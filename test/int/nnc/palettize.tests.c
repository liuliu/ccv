#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include "3rdparty/dsfmt/dSFMT.h"
#include "3rdparty/sqlite3/sqlite3.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

static void _test_iq1_write_bits(uint8_t* const data, const size_t bit, const unsigned value, const int width)
{
	int i;
	for (i = 0; i < width; i++)
		data[(bit + i) / 8] |= ((value >> i) & 1) << ((bit + i) % 8);
}

TEST_CASE("CPU IQ1 rowwise-x decodes independent packed groups across row boundaries")
{
	enum { rows = 9, cols = 37, count = rows * cols };
	const int formats[2] = {CCV_NNC_QX_8I_ROWWISE_IQ1_S, CCV_NNC_QX_8I_ROWWISE_IQ1_M};
	const int indices[4] = {0, 1, 2, 2047};
	const int scales[8] = {1, 2, 3, 5, 7, 9, 12, 16};
	int f;
	for (f = 0; f < 2; f++)
	{
		const int group_size = f == 0 ? 32 : 16;
		const int group_bits = f == 0 ? 50 : 28;
		const int groups_per_row = (cols + group_size - 1) / group_size;
		const size_t payload = (rows * groups_per_row * group_bits + 7) / 8;
		const size_t scale_offset = (payload + 127) & -128;
		const size_t size = scale_offset + rows * sizeof(float);
		const ccv_nnc_tensor_param_t params = ccv_nnc_tensor_8i_rowwise_x(CPU_TENSOR_NHWC(32F, rows, cols), formats[f]);
		REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(formats[f], CCV_32F, count, cols), size, "size includes trailing row scales");
		REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(params), size, "tensor allocation uses the same group layout");
		uint8_t* const packed = cccalloc(size, 1);
		float expected[count];
		int r, g, sg, j;
		for (r = 0; r < rows; r++)
		{
			const float row_scale = (r + 1) / 32.f;
			((float*)(packed + scale_offset))[r] = row_scale;
			for (g = 0; g < groups_per_row; g++)
			{
				const int group = r * groups_per_row + g;
				const size_t bit = (size_t)group * group_bits;
				unsigned shifts = group & 7;
				shifts |= ((shifts ^ (shifts >> 1) ^ (shifts >> 2)) & 1) << 3;
				const int m = f == 0 ? scales[group % 8] : 16 - group % 16;
				for (sg = 0; sg < group_size / 8; sg++)
				{
					const int index = indices[(group + sg) % 4];
					const int negative_shift = (shifts >> sg) & 1;
					if (f == 0)
						_test_iq1_write_bits(packed, bit + sg * 11, index, 11);
					else
						_test_iq1_write_bits(packed, bit + sg * 12, index | (negative_shift << 11), 12);
					for (j = 0; j < 8 && g * group_size + sg * 8 + j < cols; j++)
					{
						const int trit = index == 2047 ? 1 : (index == 1 && j == 0) ? 1 : (index == 2 && j < 2) ? 0 : -1;
						const int q = ccv_clamp(m * (8 * trit + (negative_shift ? -1 : 1)), -127, 127);
						expected[r * cols + g * group_size + sg * 8 + j] = row_scale * q;
					}
				}
				if (f == 0)
				{
					_test_iq1_write_bits(packed, bit + 44, shifts & 7, 3);
					_test_iq1_write_bits(packed, bit + 47, group % 8, 3);
				} else
					_test_iq1_write_bits(packed, bit + 24, m - 1, 4);
			}
		}
		float actual[count];
		ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, formats[f], actual, count);
		REQUIRE_ARRAY_EQ(float, expected, actual, count, "grid order, shifts, multipliers, saturation and tail decoding match the format");
		ccfree(packed);
	}
}

TEST_CASE("CPU IQ1 rowwise-x supports every precision, partial groups and imatrix slices")
{
	enum { rows = 9, cols = 37, count = rows * cols, slice_count = 3, rows_per_slice = rows / slice_count };
	const int formats[2] = {CCV_NNC_QX_8I_ROWWISE_IQ1_S, CCV_NNC_QX_8I_ROWWISE_IQ1_M};
	const int datatypes[4] = {CCV_16F, CCV_16BF, CCV_32F, CCV_64F};
	float values[count], importance[cols * slice_count];
	int i, f, d;
	for (i = 0; i < count; i++)
		values[i] = i < cols ? 0 : sinf(i * 0.37f) + cosf(i * 0.71f) * 0.4f;
	for (i = 0; i < cols * slice_count; i++)
		importance[i] = i % 11 == 0 ? 0 : 0.1f + (i * 17 % 31) / 8.f;
	for (d = 0; d < 4; d++)
	{
		ccv_nnc_tensor_param_t params = CPU_TENSOR_NHWC(32F, rows, cols);
		params.datatype = datatypes[d];
		ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, params, 0);
		ccv_nnc_tensor_t* const decoded = ccv_nnc_tensor_new(0, params, 0);
		ccv_nnc_tensor_t* const sliced = ccv_nnc_tensor_new(0, params, 0);
		if (datatypes[d] == CCV_16F)
			ccv_float_to_half_precision(values, (uint16_t*)source->data.u8, count);
		else if (datatypes[d] == CCV_16BF)
			ccv_float_to_bfloat(values, (uint16_t*)source->data.u8, count);
		else if (datatypes[d] == CCV_32F)
			memcpy(source->data.f32, values, sizeof(values));
		else
			for (i = 0; i < count; i++)
				source->data.f64[i] = values[i];
		const size_t element_size = CCV_GET_DATA_TYPE_SIZE(datatypes[d]);
		for (f = 0; f < 2; f++)
		{
			const size_t size = ccv_nnc_8i_rowwise_x_data_size(formats[f], datatypes[d], count, cols);
			uint8_t* const packed = ccmalloc(size + 16);
			memset(packed, 0xa5, size + 16);
			REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(source->data.u8, datatypes[d], CCV_TENSOR_CPU_MEMORY, count, cols, formats[f], importance, cols * slice_count, 0, 0, packed, size - 1), 0, "short output is rejected");
			for (i = 0; i < size + 16; i++)
				REQUIRE_EQ(packed[i], 0xa5, "rejected output stays unchanged");
			REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(source->data.u8, datatypes[d], CCV_TENSOR_CPU_MEMORY, count, cols, formats[f], importance, cols * slice_count, 0, 0, packed, size), size, "quantizer returns exact size");
			for (i = 0; i < 16; i++)
				REQUIRE_EQ(packed[size + i], 0xa5, "quantizer preserves output guard bytes");
			ccv_nnc_dequantize_8i_rowwise_x(packed, datatypes[d], CCV_TENSOR_CPU_MEMORY, size, cols, formats[f], decoded->data.u8, count);
			int slice;
			for (slice = 0; slice < slice_count; slice++)
			{
				const size_t slice_size = ccv_nnc_8i_rowwise_x_data_size(formats[f], datatypes[d], rows_per_slice * cols, cols);
				uint8_t* const slice_packed = ccmalloc(slice_size);
				const size_t offset = (size_t)slice * rows_per_slice * cols * element_size;
				REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(source->data.u8 + offset, datatypes[d], CCV_TENSOR_CPU_MEMORY, rows_per_slice * cols, cols, formats[f], importance + slice * cols, cols, 0, 0, slice_packed, slice_size), slice_size, "individual imatrix slice quantizes");
				ccv_nnc_dequantize_8i_rowwise_x(slice_packed, datatypes[d], CCV_TENSOR_CPU_MEMORY, slice_size, cols, formats[f], sliced->data.u8 + offset, rows_per_slice * cols);
				ccfree(slice_packed);
			}
			REQUIRE_ARRAY_EQ(uint8_t, decoded->data.u8, sliced->data.u8, count * element_size, "packed imatrix slices match independent quantization");
			float output[count];
			if (datatypes[d] == CCV_16F)
				ccv_half_precision_to_float((const uint16_t*)decoded->data.u8, output, count);
			else if (datatypes[d] == CCV_16BF)
				ccv_bfloat_to_float((const uint16_t*)decoded->data.u8, output, count);
			else if (datatypes[d] == CCV_32F)
				memcpy(output, decoded->data.f32, sizeof(output));
			else
				for (i = 0; i < count; i++)
					output[i] = decoded->data.f64[i];
			for (i = 0; i < count; i++)
				REQUIRE(isfinite(output[i]), "decoded values are finite");
			for (i = 0; i < cols; i++)
				REQUIRE_EQ(output[i], 0, "zero source row is exactly zero");
			ccfree(packed);
		}
		ccv_nnc_tensor_free(sliced);
		ccv_nnc_tensor_free(decoded);
		ccv_nnc_tensor_free(source);
	}
}

TEST_CASE("CPU IQ1 rowwise-x metadata and payload survive tensor persistence")
{
	const int formats[2] = {CCV_NNC_QX_8I_ROWWISE_IQ1_S, CCV_NNC_QX_8I_ROWWISE_IQ1_M};
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 37), 0);
	int i, f;
	for (i = 0; i < 3 * 37; i++)
		source->data.f32[i] = sinf(i * 0.37f);
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	for (f = 0; f < 2; f++)
	{
		ccv_nnc_tensor_t* const packed = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(source->info, formats[f]), 0);
		const size_t size = ccv_nnc_tensor_data_size_without_padding(packed->info);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(source->data.f32, CCV_32F, CCV_TENSOR_CPU_MEMORY, 3 * 37, 37, formats[f], 0, 0, 0, 0, packed->data.u8, size), size, "allocated tensor quantizes");
		const char* const name = f == 0 ? "iq1_s" : "iq1_m";
		REQUIRE_EQ(ccv_nnc_tensor_write(packed, db, name, 0), CCV_IO_FINAL, "IQ1 tensor writes");
		ccv_nnc_tensor_t* restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_FINAL, "IQ1 tensor reads");
		REQUIRE_EQ(restored->info.datatype, packed->info.datatype, "source precision persists");
		REQUIRE_EQ(restored->info.reserved, formats[f], "codec persists");
		REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(restored->info), size, "size persists");
		REQUIRE_EQ(memcmp(restored->data.u8, packed->data.u8, size), 0, "payload and trailing scales persist");
		ccv_nnc_tensor_free(restored);
		ccv_nnc_tensor_free(packed);
	}
	sqlite3_close(db);
	ccv_nnc_tensor_free(source);
}

TEST_CASE("CPU IQ1 rowwise-x fits the imatrix-weighted objective")
{
	const int formats[2] = {CCV_NNC_QX_8I_ROWWISE_IQ1_S, CCV_NNC_QX_8I_ROWWISE_IQ1_M};
	float source[32], importance[32] = {1};
	int i, f;
	for (i = 0; i < 32; i++)
		source[i] = i == 0 ? 0.317f : sinf(i * 1.32f) * 3;
	for (f = 0; f < 2; f++)
	{
		const size_t size = ccv_nnc_8i_rowwise_x_data_size(formats[f], CCV_32F, 32, 32);
		uint8_t* const packed = ccmalloc(size);
		float weighted[32], plain[32];
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(source, CCV_32F, CCV_TENSOR_CPU_MEMORY, 32, 32, formats[f], 0, 0, 0, 0, packed, size), size, "null imatrix is supported");
		ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, 32, formats[f], plain, 32);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(source, CCV_32F, CCV_TENSOR_CPU_MEMORY, 32, 32, formats[f], importance, 32, 0, 0, packed, size), size, "weighted quantization succeeds");
		ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, 32, formats[f], weighted, 32);
		REQUIRE(fabsf(weighted[0] - source[0]) < 1e-6f, "important lane is fitted accurately");
		REQUIRE(fabsf(weighted[0] - source[0]) < fabsf(plain[0] - source[0]), "imatrix changes the fitting objective");
		ccfree(packed);
	}
}

static double _test_8i_rowwise_x_format(const int format, const ccv_nnc_tensor_param_t params, int* const finite, int* const size_matches)
{
	enum {
		rows = 3,
		cols = 19,
	};
	float values[rows * cols];
	float imatrix[cols];
	int i;
	for (i = 0; i < rows * cols; i++)
		values[i] = (float)(((i * 37) % 101) - 50) / 32;
	for (i = 0; i < cols; i++)
		imatrix[i] = 0.25f + (float)(i % 7) * 0.125f;
	const size_t data_size = ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, rows * cols, cols);
	const size_t tensor_data_size = ccv_nnc_tensor_data_size_without_padding(params);
	uint8_t* const compressed = (uint8_t*)ccmalloc(data_size);
	const size_t output_size = ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, rows * cols, cols, format, imatrix, cols, 0, 0, compressed, data_size);
	*size_matches = (data_size == output_size && data_size == tensor_data_size);
	float dequantized[rows * cols];
	ccv_nnc_dequantize_8i_rowwise_x(compressed, CCV_32F, CCV_TENSOR_CPU_MEMORY, output_size, cols, format, dequantized, rows * cols);
	double mse = 0;
	*finite = 1;
	for (i = 0; i < rows * cols; i++)
	{
		*finite = *finite && isfinite(dequantized[i]);
		const double d = values[i] - dequantized[i];
		mse += d * d;
	}
	mse /= (double)(rows * cols);
	ccfree(compressed);
	return mse;
}

TEST_CASE("H256 rowwise-x quantization and dequantization match independent dense rotations for every codec and precision")
{
	enum { rows = 3, cols = 512, count = rows * cols };
	float values[count], rotated[count], imatrix[count];
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 256);
	int i, j, k, shift;
	for (i = 0; i < count; i++)
	{
		// Dyadic values and their transforms are exact in all four precisions.
		values[i] = dsfmt_genrand_open_close(&dsfmt) > 0.5 ? 1 : -1;
		imatrix[i] = 0.25f + (i % 7) * 0.125f;
	}
	for (i = 0; i < count; i += 256)
		for (j = 0; j < 256; j++)
		{
			double sum = 0;
			for (k = 0; k < 256; k++)
			{
				int sign = 1;
				for (shift = 0; shift < 8; shift += 2)
					if (((j >> shift) & 3) + ((k >> shift) & 3) == 3)
						sign = -sign;
				sum += sign * values[i + k];
			}
			rotated[i + j] = sum / 16;
		}
	const int datatypes[] = {CCV_16F, CCV_16BF, CCV_32F, CCV_64F};
	int d, codec;
	for (d = 0; d < 4; d++)
	{
		ccv_nnc_tensor_param_t params = CPU_TENSOR_NHWC(32F, rows, cols);
		params.datatype = datatypes[d];
		ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, params, 0);
		ccv_nnc_tensor_t* const reference = ccv_nnc_tensor_new(0, params, 0);
		if (datatypes[d] == CCV_16F)
		{
			ccv_float_to_half_precision(values, (uint16_t*)source->data.u8, count);
			ccv_float_to_half_precision(rotated, (uint16_t*)reference->data.u8, count);
		} else if (datatypes[d] == CCV_16BF) {
			ccv_float_to_bfloat(values, (uint16_t*)source->data.u8, count);
			ccv_float_to_bfloat(rotated, (uint16_t*)reference->data.u8, count);
		} else if (datatypes[d] == CCV_32F) {
			memcpy(source->data.f32, values, sizeof(values));
			memcpy(reference->data.f32, rotated, sizeof(rotated));
		} else {
			for (i = 0; i < count; i++)
			{
				source->data.f64[i] = values[i];
				reference->data.f64[i] = rotated[i];
			}
		}
		for (codec = CCV_NNC_QX_8I_ROWWISE_Q4_K; codec <= CCV_NNC_QX_8I_ROWWISE_Q6_K; codec++)
		{
			const int format = codec | CCV_NNC_QX_8I_ROWWISE_HADAMARD_256;
			ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(params, format), 0);
			ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(params, codec), 0);
			const size_t size = ccv_nnc_tensor_data_size_without_padding(actual->info);
			REQUIRE_EQ(actual->info.reserved, format, "tensor metadata preserves H256");
			REQUIRE_EQ(size, ccv_nnc_tensor_data_size_without_padding(expected->info), "H256 does not change storage size");
			REQUIRE_EQ(size, ccv_nnc_8i_rowwise_x_data_size(format, datatypes[d], count, cols), "byte sizing accepts modifier");
			const float* const importance = codec & 1 ? imatrix : 0;
			const size_t importance_count = importance ? count : 0;
			REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(source->data.u8, datatypes[d], CCV_TENSOR_CPU_MEMORY, count, cols, format, importance, importance_count, 0, 0, actual->data.u8, size), size, "flagged weights quantize");
			REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(reference->data.u8, datatypes[d], CCV_TENSOR_CPU_MEMORY, count, cols, codec, importance, importance_count, 0, 0, expected->data.u8, size), size, "independently rotated weights quantize");
			REQUIRE_EQ(memcmp(actual->data.u8, expected->data.u8, size), 0, "H256 must run before scale fitting and packing, codec=%d datatype=%d", codec, datatypes[d]);
			// Decode the same payload to double with its stored row scales. This
			// avoids an intermediate FP16/BF16 rounding in the independent inverse.
			const size_t payload = size - rows * CCV_GET_DATA_TYPE_SIZE(datatypes[d]);
			const size_t double_size = payload + rows * sizeof(double);
			uint8_t* const double_packed = (uint8_t*)ccmalloc(double_size);
			memcpy(double_packed, actual->data.u8, payload);
			for (i = 0; i < rows; i++)
			{
				float scale;
				if (datatypes[d] == CCV_16F)
					ccv_half_precision_to_float((uint16_t*)(actual->data.u8 + payload) + i, &scale, 1);
				else if (datatypes[d] == CCV_16BF)
					ccv_bfloat_to_float((uint16_t*)(actual->data.u8 + payload) + i, &scale, 1);
				else if (datatypes[d] == CCV_32F)
					scale = ((float*)(actual->data.u8 + payload))[i];
				((double*)(double_packed + payload))[i] = datatypes[d] == CCV_64F ? ((double*)(actual->data.u8 + payload))[i] : scale;
			}
			double decoded[count], inverse[count];
			ccv_nnc_dequantize_8i_rowwise_x(double_packed, CCV_64F, CCV_TENSOR_CPU_MEMORY, double_size, cols, codec, decoded, count);
			for (i = 0; i < count; i += 256)
				for (j = 0; j < 256; j++)
				{
					double sum = 0;
					for (k = 0; k < 256; k++)
					{
						int sign = 1;
						for (shift = 0; shift < 8; shift += 2)
							if (((j >> shift) & 3) + ((k >> shift) & 3) == 3)
								sign = -sign;
						sum += sign * decoded[i + k];
					}
					inverse[i + j] = sum / 16;
				}
			ccv_nnc_tensor_t* const logical = ccv_nnc_tensor_new(0, params, 0);
			ccv_nnc_dequantize_8i_rowwise_x(actual->data.u8, datatypes[d], CCV_TENSOR_CPU_MEMORY, size, cols, format, logical->data.u8, count);
			float logical_f32[count];
			if (datatypes[d] == CCV_16F)
				ccv_half_precision_to_float(logical->data.f16, logical_f32, count);
			else if (datatypes[d] == CCV_16BF)
				ccv_bfloat_to_float(logical->data.f16, logical_f32, count);
			else if (datatypes[d] == CCV_32F)
				memcpy(logical_f32, logical->data.f32, sizeof(logical_f32));
			for (i = 0; i < count; i++)
			{
				const double value = datatypes[d] == CCV_64F ? logical->data.f64[i] : logical_f32[i];
				const double tolerance = datatypes[d] == CCV_16F ? 1e-3 : datatypes[d] == CCV_16BF ? 8e-3 : datatypes[d] == CCV_32F ? 1e-6 : 1e-12;
				REQUIRE(fabs(value - inverse[i]) <= tolerance * ccv_max(1., fabs(inverse[i])), "logical dequantization matches independent inverse: codec=%d datatype=%d index=%d", codec, datatypes[d], i);
			}
			ccv_nnc_tensor_free(logical);
			ccfree(double_packed);
			ccv_nnc_tensor_free(actual);
			ccv_nnc_tensor_free(expected);
		}
		ccv_nnc_tensor_free(source);
		ccv_nnc_tensor_free(reference);
	}
}

TEST_CASE("H256 rowwise-x rejects invalid K without modifying output")
{
	float values[1024] = {0};
	uint8_t output[4096], untouched[4096];
	memset(output, 0xa5, sizeof(output));
	memcpy(untouched, output, sizeof(output));
	const int format = CCV_NNC_QX_8I_ROWWISE_Q6_K | CCV_NNC_QX_8I_ROWWISE_HADAMARD_256;
	const size_t invalid_k[] = {0, 1, 255, 257, 384, 513};
	int i;
	for (i = 0; i < sizeof(invalid_k) / sizeof(invalid_k[0]); i++)
	{
		REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, invalid_k[i], invalid_k[i]), 0, "invalid K has no encoded size");
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, invalid_k[i], invalid_k[i], format, 0, 0, 0, 0, output, sizeof(output)), 0, "invalid K must fail");
	}
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 513, 256, format, 0, 0, 0, 0, output, sizeof(output)), 0, "partial rows must fail");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 512, 256, format, 0, 0, 0, 0, output, 1), 0, "short output must fail");
	REQUIRE_EQ(memcmp(output, untouched, sizeof(output)), 0, "failure must not write output");
}

TEST_CASE("H256 rowwise-x metadata and payload survive tensor persistence")
{
	const int format = CCV_NNC_QX_8I_ROWWISE_Q6_K | CCV_NNC_QX_8I_ROWWISE_HADAMARD_256;
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 256), 0);
	ccv_nnc_tensor_zero(source);
	ccv_nnc_tensor_t* const packed = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(source->info, format), 0);
	const size_t size = ccv_nnc_tensor_data_size_without_padding(packed->info);
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(source->data.f32, CCV_32F, CCV_TENSOR_CPU_MEMORY, 3 * 256, 256, format, 0, 0, 0, 0, packed->data.u8, size), size, "zero weights quantize");
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	REQUIRE_EQ(ccv_nnc_tensor_write(packed, db, "h256", 0), CCV_IO_FINAL, "flagged tensor writes");
	ccv_nnc_tensor_t* restored = 0;
	REQUIRE_EQ(ccv_nnc_tensor_read(db, "h256", 0, 0, 0, &restored), CCV_IO_FINAL, "flagged tensor reads");
	REQUIRE_EQ(restored->info.reserved, format, "codec and rotation flag both persist");
	REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(restored->info), size, "size persists");
	REQUIRE_EQ(memcmp(restored->data.u8, packed->data.u8, size), 0, "payload persists");
	sqlite3_close(db);
	ccv_nnc_tensor_free(restored);
	ccv_nnc_tensor_free(packed);
	ccv_nnc_tensor_free(source);
}

TEST_CASE("allocate row-wise int8 tensor with source-precision scales")
{
	ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(32F, 10, 20, 30)), 0);
	REQUIRE_EQ(6848, ccv_nnc_tensor_data_size(tensor->info), "should be this size");
	ccv_nnc_tensor_free(tensor);
}

TEST_CASE("quantize float to row-wise int8 and dequantize on CPU losslessly")
{
	float values[32];
	static const int8_t q[8] = {-127, -96, -64, -32, 0, 32, 64, 127};
	static const float scales[4] = {0.5, 1.0, 2.0, 4.0};
	int i, j;
	for (i = 0; i < 4; i++)
		for (j = 0; j < 8; j++)
			values[i * 8 + j] = q[j] * scales[i];
	ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(32F, 4, 8)), 0);
	const size_t output_size = ccv_nnc_quantize_8i_rowwise(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 32, 8, 0, 0, tensor->data.u8, ccv_nnc_tensor_data_size_without_padding(tensor->info));
	REQUIRE_EQ(144, output_size, "output size should match");
	float dequantized[32];
	ccv_nnc_dequantize_8i_rowwise(tensor->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, output_size, 8, dequantized, 32);
	REQUIRE_ARRAY_EQ(float, values, dequantized, 32, "should be lossless");
	ccv_nnc_tensor_free(tensor);
}

TEST_CASE("quantize float to row-wise int8 with lower MSE than absmax scale")
{
	float values[8] = {
		-1.442847, 2.885900, 0.940176, 1.100095,
		-1.533835, -0.690122, -0.371144, -0.208764
	};
	ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(32F, 1, 8)), 0);
	const size_t output_size = ccv_nnc_quantize_8i_rowwise(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 8, 8, 0, 0,
		tensor->data.u8, ccv_nnc_tensor_data_size_without_padding(tensor->info));
	float dequantized[8];
	ccv_nnc_dequantize_8i_rowwise(tensor->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, output_size, 8, dequantized, 8);
	const float absmax_scale = 2.885900 / 127;
	double absmax_sse = 0;
	double rowwise_sse = 0;
	int i;
	for (i = 0; i < 8; i++)
	{
		const int q = ccv_clamp((int)lrint(values[i] / absmax_scale), -127, 127);
		const double d0 = values[i] - absmax_scale * q;
		const double d1 = values[i] - dequantized[i];
		absmax_sse += d0 * d0;
		rowwise_sse += d1 * d1;
	}
	REQUIRE(rowwise_sse < absmax_sse * 0.2, "least-squares row scale should reduce MSE");
	ccv_nnc_tensor_free(tensor);
}

TEST_CASE("quantize float to row-wise int8 with imatrix-weighted scale")
{
	float values[8] = {
		-49.257720, 0.349063, 0.214409, -0.305378,
		-0.365815, 0.553977, -0.809149, 0.585139
	};
	float imatrix[8] = {0, 1, 1, 1, 1, 1, 1, 1};
	ccv_nnc_tensor_t* const unweighted_tensor = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(32F, 1, 8)), 0);
	ccv_nnc_tensor_t* const weighted_tensor = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(32F, 1, 8)), 0);
	const size_t unweighted_size = ccv_nnc_quantize_8i_rowwise(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 8, 8, 0, 0,
		unweighted_tensor->data.u8, ccv_nnc_tensor_data_size_without_padding(unweighted_tensor->info));
	const size_t weighted_size = ccv_nnc_quantize_8i_rowwise(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 8, 8, imatrix, 8,
		weighted_tensor->data.u8, ccv_nnc_tensor_data_size_without_padding(weighted_tensor->info));
	float unweighted[8];
	float weighted[8];
	ccv_nnc_dequantize_8i_rowwise(unweighted_tensor->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, unweighted_size, 8, unweighted, 8);
	ccv_nnc_dequantize_8i_rowwise(weighted_tensor->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, weighted_size, 8, weighted, 8);
	double unweighted_sse = 0;
	double weighted_sse = 0;
	int i;
	for (i = 0; i < 8; i++)
	{
		const double d0 = values[i] - unweighted[i];
		const double d1 = values[i] - weighted[i];
		unweighted_sse += imatrix[i] * d0 * d0;
		weighted_sse += imatrix[i] * d1 * d1;
	}
	REQUIRE(weighted_sse < unweighted_sse * 0.25, "imatrix-weighted row scale should reduce weighted MSE");
	ccv_nnc_tensor_free(unweighted_tensor);
	ccv_nnc_tensor_free(weighted_tensor);
}

TEST_CASE("quantize float to row-wise int8 with packed imatrix slices")
{
	float values[16] = {
		-49.257720, 0.349063, 0.214409, -0.305378,
		-0.365815, 0.553977, -0.809149, 0.585139,
		-49.257720, 0.349063, 0.214409, -0.305378,
		-0.365815, 0.553977, -0.809149, 0.585139
	};
	float imatrix[16] = {
		1, 1, 1, 1, 1, 1, 1, 1,
		0, 1, 1, 1, 1, 1, 1, 1
	};
	ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(32F, 2, 8)), 0);
	const size_t output_size = ccv_nnc_quantize_8i_rowwise(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 16, 8, imatrix, 16,
		tensor->data.u8, ccv_nnc_tensor_data_size_without_padding(tensor->info));
	REQUIRE_EQ(output_size, ccv_nnc_tensor_data_size_without_padding(tensor->info), "packed imatrix quantization should fit");
	float dequantized[16];
	ccv_nnc_dequantize_8i_rowwise(tensor->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, output_size, 8, dequantized, 16);
	double first_row_sse = 0;
	double second_row_sse = 0;
	int i;
	for (i = 1; i < 8; i++)
	{
		const double d0 = values[i] - dequantized[i];
		const double d1 = values[8 + i] - dequantized[8 + i];
		first_row_sse += d0 * d0;
		second_row_sse += d1 * d1;
	}
	REQUIRE(second_row_sse < first_row_sse * 0.25, "second row should use the second imatrix slice");
	ccv_nnc_tensor_free(tensor);
}

TEST_CASE("quantize float to row-wise-x int8 formats and dequantize on CPU")
{
	const ccv_nnc_tensor_param_t source_params = CPU_TENSOR_NHWC(32F, 3, 19);
	const int formats[] = {
		CCV_NNC_QX_8I_ROWWISE_Q5_K,
		CCV_NNC_QX_8I_ROWWISE_Q4_K,
		CCV_NNC_QX_8I_ROWWISE_Q3_K,
		CCV_NNC_QX_8I_ROWWISE_Q2_K,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XXS,
		CCV_NNC_QX_8I_ROWWISE_IQ2_S,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XS,
		CCV_NNC_QX_8I_ROWWISE_IQ3_S,
		CCV_NNC_QX_8I_ROWWISE_IQ3_XXS,
	};
	const ccv_nnc_tensor_param_t params[] = {
		ccv_nnc_tensor_8i_rowwise_x(source_params, CCV_NNC_QX_8I_ROWWISE_Q5_K),
		ccv_nnc_tensor_8i_rowwise_x(source_params, CCV_NNC_QX_8I_ROWWISE_Q4_K),
		ccv_nnc_tensor_8i_rowwise_x(source_params, CCV_NNC_QX_8I_ROWWISE_Q3_K),
		ccv_nnc_tensor_8i_rowwise_x(source_params, CCV_NNC_QX_8I_ROWWISE_Q2_K),
		ccv_nnc_tensor_8i_rowwise_x(source_params, CCV_NNC_QX_8I_ROWWISE_IQ2_XXS),
		ccv_nnc_tensor_8i_rowwise_x(source_params, CCV_NNC_QX_8I_ROWWISE_IQ2_S),
		ccv_nnc_tensor_8i_rowwise_x(source_params, CCV_NNC_QX_8I_ROWWISE_IQ2_XS),
		ccv_nnc_tensor_8i_rowwise_x(source_params, CCV_NNC_QX_8I_ROWWISE_IQ3_S),
		ccv_nnc_tensor_8i_rowwise_x(source_params, CCV_NNC_QX_8I_ROWWISE_IQ3_XXS),
	};
	int i;
	for (i = 0; i < sizeof(formats) / sizeof(formats[0]); i++)
	{
		int finite = 0;
		int size_matches = 0;
		const double mse = _test_8i_rowwise_x_format(formats[i], params[i], &finite, &size_matches);
		REQUIRE(size_matches, "output size should match");
		REQUIRE(finite, "dequantized value should be finite");
		REQUIRE(mse < 1, "reference quantization should stay in a reasonable error range");
	}
}

static double _test_randn(dsfmt_t* const dsfmt)
{
	const double u1 = dsfmt_genrand_open_close(dsfmt);
	const double u2 = dsfmt_genrand_open_close(dsfmt);
	return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
}

// AR(1) correlated inputs with per-column scales; returns E[x x^T] over samples.
static void _test_correlated_hessian(dsfmt_t* const dsfmt, const int cols, const int samples, const double rho, float* const hessian)
{
	double* const sigma = (double*)ccmalloc(sizeof(double) * cols);
	double* const x = (double*)ccmalloc(sizeof(double) * cols);
	double* const sum = (double*)cccalloc(cols * cols, sizeof(double));
	int i, a, b;
	for (a = 0; a < cols; a++)
		sigma[a] = exp(0.5 * _test_randn(dsfmt));
	for (i = 0; i < samples; i++)
	{
		double v = _test_randn(dsfmt);
		for (a = 0; a < cols; a++)
		{
			if (a > 0)
				v = rho * v + sqrt(1 - rho * rho) * _test_randn(dsfmt);
			x[a] = v * sigma[a];
		}
		for (a = 0; a < cols; a++)
			for (b = 0; b < cols; b++)
				sum[a * cols + b] += x[a] * x[b];
	}
	for (a = 0; a < cols * cols; a++)
		hessian[a] = (float)(sum[a] / samples);
	ccfree(sum);
	ccfree(x);
	ccfree(sigma);
}

static double _test_hessian_error(const float* const w, const float* const wq, const float* const hessian, const int rows, const int cols)
{
	double total = 0;
	int i, a, b;
	for (i = 0; i < rows; i++)
		for (a = 0; a < cols; a++)
		{
			const double ea = (double)w[i * cols + a] - wq[i * cols + a];
			double v = 0;
			for (b = 0; b < cols; b++)
				v += hessian[a * cols + b] * ((double)w[i * cols + b] - wq[i * cols + b]);
			total += ea * v;
		}
	return total;
}

TEST_CASE("row-wise-x Hessian error feedback lowers the Hessian-weighted error for every codec")
{
	enum { rows = 8, cols = 256, count = rows * cols };
	float* const hessian = (float*)ccmalloc(sizeof(float) * cols * cols);
	float* const values = (float*)ccmalloc(sizeof(float) * count);
	float* const decoded = (float*)ccmalloc(sizeof(float) * count);
	float imatrix[cols];
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 3);
	_test_correlated_hessian(&dsfmt, cols, 1024, 0.9, hessian);
	int i;
	for (i = 0; i < cols; i++)
		imatrix[i] = hessian[i * cols + i];
	for (i = 0; i < count; i++)
	{
		const double u = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
		const double v = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
		values[i] = (float)(u * v * v * 0.05);
	}
	int codec;
	for (codec = CCV_NNC_QX_8I_ROWWISE_Q4_K; codec <= CCV_NNC_QX_8I_ROWWISE_IQ1_M; codec++)
	{
		const size_t size = ccv_nnc_8i_rowwise_x_data_size(codec, CCV_32F, count, cols);
		uint8_t* const packed = (uint8_t*)ccmalloc(size);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, codec, imatrix, cols, 0, 0, packed, size), size, "imatrix quantization succeeds");
		ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, codec, decoded, count);
		const double imatrix_error = _test_hessian_error(values, decoded, hessian, rows, cols);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, codec, 0, 0, hessian, cols * cols, packed, size), size, "Hessian quantization succeeds");
		ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, codec, decoded, count);
		const double hessian_error = _test_hessian_error(values, decoded, hessian, rows, cols);
		REQUIRE(hessian_error < imatrix_error * 0.9, "error feedback should lower e H e^T: codec=%d imatrix=%g hessian=%g", codec, imatrix_error, hessian_error);
		ccfree(packed);
	}
	ccfree(decoded);
	ccfree(values);
	ccfree(hessian);
}

TEST_CASE("row-wise-x diagonal Hessian matches the imatrix path")
{
	enum { rows = 8, cols = 99, count = rows * cols };
	float* const hessian = (float*)cccalloc(cols * cols, sizeof(float));
	float values[count], decoded[count], imatrix[cols];
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 4);
	int i;
	double mean = 0;
	for (i = 0; i < cols; i++)
	{
		hessian[i * cols + i] = (float)(0.05 + 4 * dsfmt_genrand_open_close(&dsfmt));
		mean += hessian[i * cols + i];
	}
	mean /= cols;
	// The Hessian path adds a 1e-6 ridge for positive definiteness.
	for (i = 0; i < cols; i++)
		imatrix[i] = (float)(hessian[i * cols + i] + mean * 1e-6);
	for (i = 0; i < count; i++)
		values[i] = (float)(_test_randn(&dsfmt) * 0.02);
	int codec;
	for (codec = CCV_NNC_QX_8I_ROWWISE_Q4_K; codec <= CCV_NNC_QX_8I_ROWWISE_IQ1_M; codec++)
	{
		const size_t size = ccv_nnc_8i_rowwise_x_data_size(codec, CCV_32F, count, cols);
		uint8_t* const packed = (uint8_t*)ccmalloc(size);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, codec, imatrix, cols, 0, 0, packed, size), size, "imatrix quantization succeeds");
		ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, codec, decoded, count);
		const double imatrix_error = _test_hessian_error(values, decoded, hessian, rows, cols);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, codec, 0, 0, hessian, cols * cols, packed, size), size, "Hessian quantization succeeds");
		ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, codec, decoded, count);
		const double hessian_error = _test_hessian_error(values, decoded, hessian, rows, cols);
		REQUIRE(fabs(hessian_error - imatrix_error) <= imatrix_error * 1e-6, "diagonal Hessian has no feedback: codec=%d imatrix=%g hessian=%g", codec, imatrix_error, hessian_error);
		ccfree(packed);
	}
	ccfree(hessian);
}

TEST_CASE("row-wise-x Hessian slices quantize rows like independent calls")
{
	// 5 rows per slice so packing chunks straddle slices; 19 columns so the last group is padded.
	enum { slice_rows = 5, rows = 2 * slice_rows, cols = 19, count = rows * cols };
	float hessian[2 * cols * cols], values[count], whole[count], sliced[count];
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 5);
	_test_correlated_hessian(&dsfmt, cols, 256, 0.8, hessian);
	_test_correlated_hessian(&dsfmt, cols, 256, 0.5, hessian + cols * cols);
	int i;
	for (i = 0; i < count; i++)
		values[i] = (float)(_test_randn(&dsfmt) * 0.02);
	const int codecs[] = {CCV_NNC_QX_8I_ROWWISE_Q2_K, CCV_NNC_QX_8I_ROWWISE_IQ2_XXS, CCV_NNC_QX_8I_ROWWISE_IQ3_XXS};
	int c;
	for (c = 0; c < 3; c++)
	{
		const size_t size = ccv_nnc_8i_rowwise_x_data_size(codecs[c], CCV_32F, count, cols);
		uint8_t* const packed = (uint8_t*)ccmalloc(size);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, codecs[c], 0, 0, hessian, 2 * cols * cols, packed, size), size, "sliced Hessian quantization succeeds");
		ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, codecs[c], whole, count);
		ccfree(packed);
		int s;
		for (s = 0; s < 2; s++)
		{
			const size_t slice_size = ccv_nnc_8i_rowwise_x_data_size(codecs[c], CCV_32F, slice_rows * cols, cols);
			uint8_t* const slice_packed = (uint8_t*)ccmalloc(slice_size);
			REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values + s * slice_rows * cols, CCV_32F, CCV_TENSOR_CPU_MEMORY, slice_rows * cols, cols, codecs[c], 0, 0, hessian + s * cols * cols, cols * cols, slice_packed, slice_size), slice_size, "independent slice quantization succeeds");
			ccv_nnc_dequantize_8i_rowwise_x(slice_packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, slice_size, cols, codecs[c], sliced + s * slice_rows * cols, slice_rows * cols);
			ccfree(slice_packed);
		}
		REQUIRE_ARRAY_EQ(float, whole, sliced, count, "Hessian slices must match independent quantization: codec=%d", codecs[c]);
	}
}

TEST_CASE("row-wise-x Hessian rejects unsupported input and survives a singular Hessian")
{
	enum { rows = 4, cols = 256, count = rows * cols };
	float* const hessian = (float*)cccalloc(cols * cols, sizeof(float));
	float values[count], decoded[count], imatrix[cols];
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 6);
	int i, j;
	for (i = 0; i < count; i++)
		values[i] = (float)(_test_randn(&dsfmt) * 0.02);
	for (i = 0; i < cols; i++)
	{
		hessian[i * cols + i] = 1;
		imatrix[i] = 1;
	}
	const int codec = CCV_NNC_QX_8I_ROWWISE_IQ2_XXS;
	const size_t size = ccv_nnc_8i_rowwise_x_data_size(codec, CCV_32F, count, cols);
	uint8_t* const packed = (uint8_t*)ccmalloc(size + 16);
	memset(packed, 0xa5, size + 16);
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, codec | CCV_NNC_QX_8I_ROWWISE_HADAMARD_256, 0, 0, hessian, cols * cols, packed, size), 0, "H256 is not supported with a Hessian");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, codec, 0, 0, hessian, cols * cols - 1, packed, size), 0, "partial Hessian must fail");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 3 * cols, cols, codec, 0, 0, hessian, 2 * cols * cols, packed, size), 0, "rows must divide evenly into Hessian slices");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, codec, imatrix, cols, hessian, cols * cols, packed, size), 0, "imatrix and Hessian are exclusive");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, codec, 0, 0, hessian, cols * cols, packed, size - 1), 0, "short output must fail");
	for (i = 0; i < size + 16; i++)
		REQUIRE_EQ(packed[i], 0xa5, "rejected calls must not write output");
	// Rank one, then all zero: the ridge keeps both factorizable.
	for (i = 0; i < cols; i++)
		for (j = 0; j < cols; j++)
			hessian[i * cols + j] = (float)(sin(i * 0.37) * sin(j * 0.37));
	int pass;
	for (pass = 0; pass < 2; pass++)
	{
		if (pass == 1)
			memset(hessian, 0, sizeof(float) * cols * cols);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, codec, 0, 0, hessian, cols * cols, packed, size), size, "singular Hessian quantizes");
		REQUIRE_EQ(packed[size], 0xa5, "quantizer preserves output guard bytes");
		ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, codec, decoded, count);
		for (i = 0; i < count; i++)
			REQUIRE(isfinite(decoded[i]), "decoded values are finite");
	}
	ccfree(packed);
	ccfree(hessian);
}

TEST_CASE("quantize bfloat16 to row-wise int8 and dequantize on CPU losslessly")
{
	float values_f32[32];
	uint16_t values_bf16[32];
	uint16_t expected_bf16[32];
	static const int8_t q[8] = {-127, -96, -64, -32, 0, 32, 64, 127};
	static const float scales[4] = {0.5, 1.0, 2.0, 4.0};
	int i, j;
	for (i = 0; i < 4; i++)
		for (j = 0; j < 8; j++)
			values_f32[i * 8 + j] = q[j] * scales[i];
	ccv_float_to_bfloat(values_f32, values_bf16, 32);
	memcpy(expected_bf16, values_bf16, sizeof(expected_bf16));
	ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(16BF, 4, 8)), 0);
	const size_t output_size = ccv_nnc_quantize_8i_rowwise(values_bf16, CCV_16BF, CCV_TENSOR_CPU_MEMORY, 32, 8, 0, 0, tensor->data.u8, ccv_nnc_tensor_data_size_without_padding(tensor->info));
	REQUIRE_EQ(136, output_size, "output size should match");
	uint16_t dequantized[32];
	ccv_nnc_dequantize_8i_rowwise(tensor->data.u8, CCV_16BF, CCV_TENSOR_CPU_MEMORY, output_size, 8, dequantized, 32);
	REQUIRE_ARRAY_EQ(uint16_t, expected_bf16, dequantized, 32, "should be lossless");
	ccv_nnc_tensor_free(tensor);
}

TEST_CASE("quantize float to row-wise int8 and dequantize on GPU losslessly")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float values[32];
	static const int8_t q[8] = {-127, -96, -64, -32, 0, 32, 64, 127};
	static const float scales[4] = {0.5, 1.0, 2.0, 4.0};
	int i, j;
	for (i = 0; i < 4; i++)
		for (j = 0; j < 8; j++)
			values[i * 8 + j] = q[j] * scales[i];
	ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(32F, 4, 8)), 0);
	const size_t output_size = ccv_nnc_quantize_8i_rowwise(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 32, 8, 0, 0, tensor->data.u8, ccv_nnc_tensor_data_size_without_padding(tensor->info));
	ccv_nnc_tensor_t* const g_tensor = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(GPU_TENSOR_NHWC(000, 32F, 4, 8)), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* const gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 8), 0);
	ccv_nnc_dequantize_8i_rowwise(g_tensor->data.u8, CCV_32F, CCV_TENSOR_GPU_MEMORY, output_size, 8, gv_tensor->data.u8, 32);
	ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 8), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, values, v_tensor->data.f32, 32, 1e-6, "should be lossless");
	ccv_nnc_tensor_free(v_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(tensor);
}

#ifdef HAVE_CUDA
TEST_CASE("quantize and dequantize packed row-wise int8-x on CUDA for all formats and datatypes")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	const int formats[] = {
		CCV_NNC_QX_8I_ROWWISE_Q5_K,
		CCV_NNC_QX_8I_ROWWISE_Q6_K,
		CCV_NNC_QX_8I_ROWWISE_Q4_K,
		CCV_NNC_QX_8I_ROWWISE_Q3_K,
		CCV_NNC_QX_8I_ROWWISE_Q2_K,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XXS,
		CCV_NNC_QX_8I_ROWWISE_IQ2_S,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XS,
		CCV_NNC_QX_8I_ROWWISE_IQ3_S,
		CCV_NNC_QX_8I_ROWWISE_IQ3_XXS,
	};
	const int datatypes[] = {
		CCV_16F,
		CCV_16BF,
		CCV_32F,
		CCV_64F,
	};
	const int row_lengths[] = {
		128,
		130,
		131,
		132,
		1,
	};
	const int row_counts[] = {
		11,
		11,
		11,
		11,
		65536,
	};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	int s;
	for (s = 0; s < sizeof(row_lengths) / sizeof(row_lengths[0]); s++)
	{
		const int rows = row_counts[s];
		const int cols = row_lengths[s];
		const size_t count = (size_t)rows * cols;
		float* const values = ccmalloc(sizeof(float) * count);
		size_t i;
		for (i = 0; i < count; i++)
			values[i] = (float)(dsfmt_genrand_open_close(&dsfmt) * 16 - 8);
		int d;
		const int datatype_count = rows > 65535 ? 1 : sizeof(datatypes) / sizeof(datatypes[0]);
		for (d = 0; d < datatype_count; d++)
		{
			ccv_nnc_tensor_param_t dense_params = CPU_TENSOR_NHWC(32F, rows, cols);
			dense_params.datatype = datatypes[d];
			ccv_nnc_tensor_param_t gpu_dense_params = dense_params;
			gpu_dense_params.type = CCV_TENSOR_GPU_MEMORY | 000;
			ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, dense_params, 0);
			if (datatypes[d] == CCV_16F)
				ccv_float_to_half_precision(values, (uint16_t*)source->data.f16, count);
			else if (datatypes[d] == CCV_16BF)
				ccv_float_to_bfloat(values, (uint16_t*)source->data.f16, count);
			else if (datatypes[d] == CCV_32F)
				memcpy(source->data.f32, values, sizeof(float) * count);
			else {
				for (i = 0; i < count; i++)
					source->data.f64[i] = values[i];
			}
			int f;
			const int format_count = rows > 65535 ? 1 : sizeof(formats) / sizeof(formats[0]);
			for (f = 0; f < format_count; f++)
			{
				const ccv_nnc_tensor_param_t q_params = ccv_nnc_tensor_8i_rowwise_x(dense_params, formats[f]);
				ccv_nnc_tensor_param_t gpu_q_params = q_params;
				gpu_q_params.type = CCV_TENSOR_GPU_MEMORY | 000;
				ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, q_params, 0);
				const size_t qsize = ccv_nnc_quantize_8i_rowwise_x(source->data.u8, datatypes[d], CCV_TENSOR_CPU_MEMORY, count, cols, formats[f], 0, 0, 0, 0, q->data.u8, ccv_nnc_tensor_data_size_without_padding(q->info));
				REQUIRE_EQ(qsize, ccv_nnc_tensor_data_size_without_padding(q->info), "packed row-wise int8-x size should match");
				ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, dense_params, 0);
				ccv_nnc_dequantize_8i_rowwise_x(q->data.u8, datatypes[d], CCV_TENSOR_CPU_MEMORY, qsize, cols, formats[f], expected->data.u8, count);
				ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, gpu_q_params, 0);
				ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, gpu_dense_params, 0);
				ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, dense_params, 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q), TENSOR_LIST(gq), 0);
				ccv_nnc_dequantize_8i_rowwise_x(gq->data.u8, datatypes[d], CCV_TENSOR_GPU_MEMORY, qsize, cols, formats[f], gout->data.u8, count);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(actual), 0);
				if (datatypes[d] == CCV_16F || datatypes[d] == CCV_16BF) {
					float* const expected_f32 = ccmalloc(sizeof(float) * count);
					float* const actual_f32 = ccmalloc(sizeof(float) * count);
					if (datatypes[d] == CCV_16F)
					{
						ccv_half_precision_to_float((uint16_t*)expected->data.f16, expected_f32, count);
						ccv_half_precision_to_float((uint16_t*)actual->data.f16, actual_f32, count);
						REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_f32, actual_f32, count, 1e-2, "CUDA packed row-wise int8-x half output should match CPU");
					} else {
						ccv_bfloat_to_float((uint16_t*)expected->data.f16, expected_f32, count);
						ccv_bfloat_to_float((uint16_t*)actual->data.f16, actual_f32, count);
						REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_f32, actual_f32, count, 1e-1, "CUDA packed row-wise int8-x bfloat output should match CPU");
					}
					ccfree(actual_f32);
					ccfree(expected_f32);
				} else if (datatypes[d] == CCV_32F) {
					REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected->data.f32, actual->data.f32, count, 1e-6, "CUDA packed row-wise int8-x output should match CPU");
				} else {
					REQUIRE_ARRAY_EQ_WITH_TOLERANCE(double, expected->data.f64, actual->data.f64, count, 1e-12, "CUDA packed row-wise int8-x output should match CPU");
				}
				ccv_nnc_tensor_free(actual);
				ccv_nnc_tensor_free(gout);
				ccv_nnc_tensor_free(gq);
				ccv_nnc_tensor_free(expected);
				ccv_nnc_tensor_free(q);
			}
			ccv_nnc_tensor_free(source);
		}
		ccfree(values);
	}
}
#endif

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
