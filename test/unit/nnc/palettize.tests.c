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

TEST_CASE("quantize double to 4-bit and dequantize on CPU losslessly")
{
	double lut[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
	double* const values = ccmalloc(sizeof(double) * 2839);
	int i;
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 16];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (1420 + 2944));
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2839, 4, 128, compressed, 1420 + 2944);
	REQUIRE_EQ(output_size, 1420 + 2944, "output size should match");
	double* const output_values = ccmalloc(sizeof(double) * 2839);
	ccv_nnc_depalettize(compressed, CCV_64F, CCV_TENSOR_CPU_MEMORY, output_size, 4, 128, output_values, 2839);
	REQUIRE_ARRAY_EQ(double, values, output_values, 2839, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize float to 4-bit and dequantize on CPU losslessly")
{
	float lut[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
	float* const values = ccmalloc(sizeof(float) * 2839);
	int i;
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 16];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (1420 + 2944 / 2));
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 2839, 4, 128, compressed, 1420 + 2944 / 2);
	REQUIRE_EQ(output_size, 1420 + 2944 / 2, "output size should match");
	float* const output_values = ccmalloc(sizeof(double) * 2839);
	ccv_nnc_depalettize(compressed, CCV_32F, CCV_TENSOR_CPU_MEMORY, output_size, 4, 128, output_values, 2839);
	REQUIRE_ARRAY_EQ(float, values, output_values, 2839, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize half-precision to 4-bit and dequantize on CPU losslessly")
{
	float lut_f32[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
	uint16_t lut[16];
	ccv_float_to_half_precision(lut_f32, lut, 16);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 2839);
	int i;
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 16];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (1420 + 2944 / 4));
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 2839, 4, 128, compressed, 1420 + 2944 / 4);
	REQUIRE_EQ(output_size, 1420 + 2944 / 4, "output size should match");
	uint16_t* const output_values = ccmalloc(sizeof(uint16_t) * 2839);
	ccv_nnc_depalettize(compressed, CCV_16F, CCV_TENSOR_CPU_MEMORY, output_size, 4, 128, output_values, 2839);
	REQUIRE_ARRAY_EQ(uint16_t, values, output_values, 2839, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize double to 5-bit and dequantize on CPU losslessly")
{
	double lut[32] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0};
	double* const values = ccmalloc(sizeof(double) * 2839);
	int i;
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 32];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (1775 + 23 * 32 * 8));
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2839, 5, 128, compressed, 1775 + 23 * 32 * 8);
	REQUIRE_EQ(output_size, 1775 + 23 * 32 * 8, "output size should match");
	double* const output_values = ccmalloc(sizeof(double) * 2839);
	ccv_nnc_depalettize(compressed, CCV_64F, CCV_TENSOR_CPU_MEMORY, output_size, 5, 128, output_values, 2839);
	REQUIRE_ARRAY_EQ(double, values, output_values, 2839, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize float to 5-bit and dequantize on CPU losslessly")
{
	float lut[32] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0};
	float* const values = ccmalloc(sizeof(float) * 2839);
	int i;
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 32];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (1775 + 23 * 32 * 4));
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 2839, 5, 128, compressed, 1775 + 23 * 32 * 4);
	REQUIRE_EQ(output_size, 1775 + 23 * 32 * 4, "output size should match");
	float* const output_values = ccmalloc(sizeof(double) * 2839);
	ccv_nnc_depalettize(compressed, CCV_32F, CCV_TENSOR_CPU_MEMORY, output_size, 5, 128, output_values, 2839);
	REQUIRE_ARRAY_EQ(float, values, output_values, 2839, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize half-precision to 5-bit and dequantize on CPU losslessly")
{
	float lut_f32[32] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0};
	uint16_t lut[32];
	ccv_float_to_half_precision(lut_f32, lut, 32);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 2839);
	int i;
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 32];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (1775 + 23 * 32 * 2));
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 2839, 5, 128, compressed, 1775 + 23 * 32 * 2);
	REQUIRE_EQ(output_size, 1775 + 23 * 32 * 2, "output size should match");
	uint16_t* const output_values = ccmalloc(sizeof(uint16_t) * 2839);
	ccv_nnc_depalettize(compressed, CCV_16F, CCV_TENSOR_CPU_MEMORY, output_size, 5, 128, output_values, 2839);
	REQUIRE_ARRAY_EQ(uint16_t, values, output_values, 2839, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize double to 6-bit and dequantize on CPU losslessly")
{
	double lut[64];
	int i;
	for (i = 0; i < 64; i++)
		lut[i] = (double)i;
	double* const values = ccmalloc(sizeof(double) * 2839);
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 64];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (2130 + 6 * 64 * 8));
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2839, 6, 512, compressed, 2130 + 6 * 64 * 8);
	REQUIRE_EQ(output_size, 2130 + 6 * 64 * 8, "output size should match");
	double* const output_values = ccmalloc(sizeof(double) * 2839);
	ccv_nnc_depalettize(compressed, CCV_64F, CCV_TENSOR_CPU_MEMORY, output_size, 6, 512, output_values, 2839);
	REQUIRE_ARRAY_EQ(double, values, output_values, 2839, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize float to 6-bit and dequantize on CPU losslessly")
{
	float lut[64];
	int i;
	for (i = 0; i < 64; i++)
		lut[i] = (float)i;
	float* const values = ccmalloc(sizeof(float) * 2839);
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 64];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (2130 + 6 * 64 * 4));
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 2839, 6, 512, compressed, 2130 + 6 * 64 * 4);
	REQUIRE_EQ(output_size, 2130 + 6 * 64 * 4, "output size should match");
	float* const output_values = ccmalloc(sizeof(float) * 2839);
	ccv_nnc_depalettize(compressed, CCV_32F, CCV_TENSOR_CPU_MEMORY, output_size, 6, 512, output_values, 2839);
	REQUIRE_ARRAY_EQ(float, values, output_values, 2839, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize half-precision to 6-bit and dequantize on CPU losslessly")
{
	float lut_f32[64];
	int i;
	for (i = 0; i < 64; i++)
		lut_f32[i] = (float)i;
	uint16_t lut[64];
	ccv_float_to_half_precision(lut_f32, lut, 64);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 2839);
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 64];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (2130 + 6 * 64 * 2));
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 2839, 6, 512, compressed, 2130 + 6 * 64 * 2);
	REQUIRE_EQ(output_size, 2130 + 6 * 64 * 2, "output size should match");
	uint16_t* const output_values = ccmalloc(sizeof(uint16_t) * 2839);
	ccv_nnc_depalettize(compressed, CCV_16F, CCV_TENSOR_CPU_MEMORY, output_size, 6, 512, output_values, 2839);
	REQUIRE_ARRAY_EQ(uint16_t, values, output_values, 2839, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize double to 7-bit and dequantize on CPU losslessly")
{
	double lut[128];
	int i;
	for (i = 0; i < 128; i++)
		lut[i] = (double)i;
	double* const values = ccmalloc(sizeof(double) * 2839);
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 128];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (2485 + 6 * 128 * 8));
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2839, 7, 512, compressed, 2485 + 6 * 128 * 8);
	REQUIRE_EQ(output_size, 2485 + 6 * 128 * 8, "output size should match");
	double* const output_values = ccmalloc(sizeof(double) * 2839);
	ccv_nnc_depalettize(compressed, CCV_64F, CCV_TENSOR_CPU_MEMORY, output_size, 7, 512, output_values, 2839);
	REQUIRE_ARRAY_EQ(double, values, output_values, 2839, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize float to 7-bit and dequantize on CPU losslessly")
{
	float lut[128];
	int i;
	for (i = 0; i < 128; i++)
		lut[i] = (float)i;
	float* const values = ccmalloc(sizeof(float) * 2839);
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 128];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (2485 + 6 * 128 * 4));
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 2839, 7, 512, compressed, 2485 + 6 * 128 * 4);
	REQUIRE_EQ(output_size, 2485 + 6 * 128 * 4, "output size should match");
	float* const output_values = ccmalloc(sizeof(float) * 2839);
	ccv_nnc_depalettize(compressed, CCV_32F, CCV_TENSOR_CPU_MEMORY, output_size, 7, 512, output_values, 2839);
	REQUIRE_ARRAY_EQ(float, values, output_values, 2839, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize half-precision to 7-bit and dequantize on CPU losslessly")
{
	float lut_f32[128];
	int i;
	for (i = 0; i < 128; i++)
		lut_f32[i] = (float)i;
	uint16_t lut[128];
	ccv_float_to_half_precision(lut_f32, lut, 128);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 2839);
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 128];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (2485 + 6 * 128 * 2));
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 2839, 7, 512, compressed, 2485 + 6 * 128 * 2);
	REQUIRE_EQ(output_size, 2485 + 6 * 128 * 2, "output size should match");
	uint16_t* const output_values = ccmalloc(sizeof(uint16_t) * 2839);
	ccv_nnc_depalettize(compressed, CCV_16F, CCV_TENSOR_CPU_MEMORY, output_size, 7, 512, output_values, 2839);
	REQUIRE_ARRAY_EQ(uint16_t, values, output_values, 2839, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize double to 8-bit and dequantize on CPU losslessly")
{
	double lut[256];
	int i;
	for (i = 0; i < 256; i++)
		lut[i] = (double)i;
	double* const values = ccmalloc(sizeof(double) * 2839);
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 256];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (2839 + 3 * 256 * 8));
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2839, 8, 1280, compressed, 2839 + 3 * 256 * 8);
	REQUIRE_EQ(output_size, 2839 + 3 * 256 * 8, "output size should match");
	double* const output_values = ccmalloc(sizeof(double) * 2839);
	ccv_nnc_depalettize(compressed, CCV_64F, CCV_TENSOR_CPU_MEMORY, output_size, 8, 1280, output_values, 2839);
	REQUIRE_ARRAY_EQ(double, values, output_values, 2839, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize float to 8-bit and dequantize on CPU losslessly")
{
	float lut[256];
	int i;
	for (i = 0; i < 256; i++)
		lut[i] = (float)i;
	float* const values = ccmalloc(sizeof(float) * 2839);
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 256];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (2839 + 3 * 256 * 4));
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 2839, 8, 1280, compressed, 2839 + 3 * 256 * 4);
	REQUIRE_EQ(output_size, 2839 + 3 * 256 * 4, "output size should match");
	float* const output_values = ccmalloc(sizeof(float) * 2839);
	ccv_nnc_depalettize(compressed, CCV_32F, CCV_TENSOR_CPU_MEMORY, output_size, 8, 1280, output_values, 2839);
	REQUIRE_ARRAY_EQ(float, values, output_values, 2839, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize half-precision to 8-bit and dequantize on CPU losslessly")
{
	float lut_f32[256];
	int i;
	for (i = 0; i < 256; i++)
		lut_f32[i] = (float)i;
	uint16_t lut[256];
	ccv_float_to_half_precision(lut_f32, lut, 256);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 2839);
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 256];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (2839 + 3 * 256 * 2));
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 2839, 8, 1280, compressed, 2839 + 3 * 256 * 2);
	REQUIRE_EQ(output_size, 2839 + 3 * 256 * 2, "output size should match");
	uint16_t* const output_values = ccmalloc(sizeof(uint16_t) * 2839);
	ccv_nnc_depalettize(compressed, CCV_16F, CCV_TENSOR_CPU_MEMORY, output_size, 8, 1280, output_values, 2839);
	REQUIRE_ARRAY_EQ(uint16_t, values, output_values, 2839, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize double to 4-bit and dequantize on CPU losslessly, fast path")
{
	double lut[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
	double* const values = ccmalloc(sizeof(double) * 2840);
	int i;
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 16];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (1420 + 2944));
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2840, 4, 128, compressed, 1420 + 2944);
	REQUIRE_EQ(output_size, 1420 + 2944, "output size should match");
	double* const output_values = ccmalloc(sizeof(double) * 2840);
	ccv_nnc_depalettize(compressed, CCV_64F, CCV_TENSOR_CPU_MEMORY, output_size, 4, 128, output_values, 2840);
	REQUIRE_ARRAY_EQ(double, values, output_values, 2840, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize float to 4-bit and dequantize on CPU losslessly, fast path")
{
	float lut[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
	float* const values = ccmalloc(sizeof(float) * 2840);
	int i;
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 16];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (1420 + 2944 / 2));
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 2840, 4, 128, compressed, 1420 + 2944 / 2);
	REQUIRE_EQ(output_size, 1420 + 2944 / 2, "output size should match");
	float* const output_values = ccmalloc(sizeof(double) * 2840);
	ccv_nnc_depalettize(compressed, CCV_32F, CCV_TENSOR_CPU_MEMORY, output_size, 4, 128, output_values, 2840);
	REQUIRE_ARRAY_EQ(float, values, output_values, 2840, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize half-precision to 4-bit and dequantize on CPU losslessly, fast path")
{
	float lut_f32[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
	uint16_t lut[16];
	ccv_float_to_half_precision(lut_f32, lut, 16);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 2840);
	int i;
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 16];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (1420 + 2944 / 4));
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 2840, 4, 128, compressed, 1420 + 2944 / 4);
	REQUIRE_EQ(output_size, 1420 + 2944 / 4, "output size should match");
	uint16_t* const output_values = ccmalloc(sizeof(uint16_t) * 2840);
	ccv_nnc_depalettize(compressed, CCV_16F, CCV_TENSOR_CPU_MEMORY, output_size, 4, 128, output_values, 2840);
	REQUIRE_ARRAY_EQ(uint16_t, values, output_values, 2840, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize double to 5-bit and dequantize on CPU losslessly, fast path")
{
	double lut[32] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0};
	double* const values = ccmalloc(sizeof(double) * 2840);
	int i;
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 32];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (1775 + 23 * 32 * 8));
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2840, 5, 128, compressed, 1775 + 23 * 32 * 8);
	REQUIRE_EQ(output_size, 1775 + 23 * 32 * 8, "output size should match");
	double* const output_values = ccmalloc(sizeof(double) * 2840);
	ccv_nnc_depalettize(compressed, CCV_64F, CCV_TENSOR_CPU_MEMORY, output_size, 5, 128, output_values, 2840);
	REQUIRE_ARRAY_EQ(double, values, output_values, 2840, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize float to 5-bit and dequantize on CPU losslessly, fast path")
{
	float lut[32] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0};
	float* const values = ccmalloc(sizeof(float) * 2840);
	int i;
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 32];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (1775 + 23 * 32 * 4));
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 2840, 5, 128, compressed, 1775 + 23 * 32 * 4);
	REQUIRE_EQ(output_size, 1775 + 23 * 32 * 4, "output size should match");
	float* const output_values = ccmalloc(sizeof(double) * 2840);
	ccv_nnc_depalettize(compressed, CCV_32F, CCV_TENSOR_CPU_MEMORY, output_size, 5, 128, output_values, 2840);
	REQUIRE_ARRAY_EQ(float, values, output_values, 2840, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize half-precision to 5-bit and dequantize on CPU losslessly, fast path")
{
	float lut_f32[32] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0};
	uint16_t lut[32];
	ccv_float_to_half_precision(lut_f32, lut, 32);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 2840);
	int i;
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 32];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (1775 + 23 * 32 * 2));
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 2840, 5, 128, compressed, 1775 + 23 * 32 * 2);
	REQUIRE_EQ(output_size, 1775 + 23 * 32 * 2, "output size should match");
	uint16_t* const output_values = ccmalloc(sizeof(uint16_t) * 2840);
	ccv_nnc_depalettize(compressed, CCV_16F, CCV_TENSOR_CPU_MEMORY, output_size, 5, 128, output_values, 2840);
	REQUIRE_ARRAY_EQ(uint16_t, values, output_values, 2840, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize double to 6-bit and dequantize on CPU losslessly, fast path")
{
	double lut[64];
	int i;
	for (i = 0; i < 64; i++)
		lut[i] = (double)i;
	double* const values = ccmalloc(sizeof(double) * 2840);
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 64];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (2130 + 6 * 64 * 8));
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2840, 6, 512, compressed, 2130 + 6 * 64 * 8);
	REQUIRE_EQ(output_size, 2130 + 6 * 64 * 8, "output size should match");
	double* const output_values = ccmalloc(sizeof(double) * 2840);
	ccv_nnc_depalettize(compressed, CCV_64F, CCV_TENSOR_CPU_MEMORY, output_size, 6, 512, output_values, 2840);
	REQUIRE_ARRAY_EQ(double, values, output_values, 2840, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize float to 6-bit and dequantize on CPU losslessly, fast path")
{
	float lut[64];
	int i;
	for (i = 0; i < 64; i++)
		lut[i] = (float)i;
	float* const values = ccmalloc(sizeof(float) * 2840);
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 64];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (2130 + 6 * 64 * 4));
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 2840, 6, 512, compressed, 2130 + 6 * 64 * 4);
	REQUIRE_EQ(output_size, 2130 + 6 * 64 * 4, "output size should match");
	float* const output_values = ccmalloc(sizeof(float) * 2840);
	ccv_nnc_depalettize(compressed, CCV_32F, CCV_TENSOR_CPU_MEMORY, output_size, 6, 512, output_values, 2840);
	REQUIRE_ARRAY_EQ(float, values, output_values, 2840, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize half-precision to 6-bit and dequantize on CPU losslessly, fast path")
{
	float lut_f32[64];
	int i;
	for (i = 0; i < 64; i++)
		lut_f32[i] = (float)i;
	uint16_t lut[64];
	ccv_float_to_half_precision(lut_f32, lut, 64);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 2840);
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 64];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (2130 + 6 * 64 * 2));
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 2840, 6, 512, compressed, 2130 + 6 * 64 * 2);
	REQUIRE_EQ(output_size, 2130 + 6 * 64 * 2, "output size should match");
	uint16_t* const output_values = ccmalloc(sizeof(uint16_t) * 2840);
	ccv_nnc_depalettize(compressed, CCV_16F, CCV_TENSOR_CPU_MEMORY, output_size, 6, 512, output_values, 2840);
	REQUIRE_ARRAY_EQ(uint16_t, values, output_values, 2840, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize double to 7-bit and dequantize on CPU losslessly, fast path")
{
	double lut[128];
	int i;
	for (i = 0; i < 128; i++)
		lut[i] = (double)i;
	double* const values = ccmalloc(sizeof(double) * 2840);
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 128];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (2485 + 6 * 128 * 8));
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2840, 7, 512, compressed, 2485 + 6 * 128 * 8);
	REQUIRE_EQ(output_size, 2485 + 6 * 128 * 8, "output size should match");
	double* const output_values = ccmalloc(sizeof(double) * 2840);
	ccv_nnc_depalettize(compressed, CCV_64F, CCV_TENSOR_CPU_MEMORY, output_size, 7, 512, output_values, 2840);
	REQUIRE_ARRAY_EQ(double, values, output_values, 2840, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize float to 7-bit and dequantize on CPU losslessly, fast path")
{
	float lut[128];
	int i;
	for (i = 0; i < 128; i++)
		lut[i] = (float)i;
	float* const values = ccmalloc(sizeof(float) * 2840);
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 128];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (2485 + 6 * 128 * 4));
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 2840, 7, 512, compressed, 2485 + 6 * 128 * 4);
	REQUIRE_EQ(output_size, 2485 + 6 * 128 * 4, "output size should match");
	float* const output_values = ccmalloc(sizeof(float) * 2840);
	ccv_nnc_depalettize(compressed, CCV_32F, CCV_TENSOR_CPU_MEMORY, output_size, 7, 512, output_values, 2840);
	REQUIRE_ARRAY_EQ(float, values, output_values, 2840, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize half-precision to 7-bit and dequantize on CPU losslessly, fast path")
{
	float lut_f32[128];
	int i;
	for (i = 0; i < 128; i++)
		lut_f32[i] = (float)i;
	uint16_t lut[128];
	ccv_float_to_half_precision(lut_f32, lut, 128);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 2840);
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 128];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (2485 + 6 * 128 * 2));
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 2840, 7, 512, compressed, 2485 + 6 * 128 * 2);
	REQUIRE_EQ(output_size, 2485 + 6 * 128 * 2, "output size should match");
	uint16_t* const output_values = ccmalloc(sizeof(uint16_t) * 2840);
	ccv_nnc_depalettize(compressed, CCV_16F, CCV_TENSOR_CPU_MEMORY, output_size, 7, 512, output_values, 2840);
	REQUIRE_ARRAY_EQ(uint16_t, values, output_values, 2840, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize double to 8-bit and dequantize on CPU losslessly, fast path")
{
	double lut[256];
	int i;
	for (i = 0; i < 256; i++)
		lut[i] = (double)i;
	double* const values = ccmalloc(sizeof(double) * 2840);
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 256];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (2840 + 3 * 256 * 8));
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2840, 8, 1280, compressed, 2840 + 3 * 256 * 8);
	REQUIRE_EQ(output_size, 2840 + 3 * 256 * 8, "output size should match");
	double* const output_values = ccmalloc(sizeof(double) * 2840);
	ccv_nnc_depalettize(compressed, CCV_64F, CCV_TENSOR_CPU_MEMORY, output_size, 8, 1280, output_values, 2840);
	REQUIRE_ARRAY_EQ(double, values, output_values, 2840, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize float to 8-bit and dequantize on CPU losslessly, fast path")
{
	float lut[256];
	int i;
	for (i = 0; i < 256; i++)
		lut[i] = (float)i;
	float* const values = ccmalloc(sizeof(float) * 2840);
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 256];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (2840 + 3 * 256 * 4));
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 2840, 8, 1280, compressed, 2840 + 3 * 256 * 4);
	REQUIRE_EQ(output_size, 2840 + 3 * 256 * 4, "output size should match");
	float* const output_values = ccmalloc(sizeof(float) * 2840);
	ccv_nnc_depalettize(compressed, CCV_32F, CCV_TENSOR_CPU_MEMORY, output_size, 8, 1280, output_values, 2840);
	REQUIRE_ARRAY_EQ(float, values, output_values, 2840, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

TEST_CASE("quantize half-precision to 8-bit and dequantize on CPU losslessly, fast path")
{
	float lut_f32[256];
	int i;
	for (i = 0; i < 256; i++)
		lut_f32[i] = (float)i;
	uint16_t lut[256];
	ccv_float_to_half_precision(lut_f32, lut, 256);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 2840);
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 256];
	uint8_t* compressed = ccmalloc(sizeof(uint8_t) * (2840 + 3 * 256 * 2));
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 2840, 8, 1280, compressed, 2840 + 3 * 256 * 2);
	REQUIRE_EQ(output_size, 2840 + 3 * 256 * 2, "output size should match");
	uint16_t* const output_values = ccmalloc(sizeof(uint16_t) * 2840);
	ccv_nnc_depalettize(compressed, CCV_16F, CCV_TENSOR_CPU_MEMORY, output_size, 8, 1280, output_values, 2840);
	REQUIRE_ARRAY_EQ(uint16_t, values, output_values, 2840, "should be lossless");
	ccfree(values);
	ccfree(output_values);
	ccfree(compressed);
}

#include "case_main.h"
