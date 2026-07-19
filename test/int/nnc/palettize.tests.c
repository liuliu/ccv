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
	const size_t output_size = ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, rows * cols, cols, format, imatrix, cols, compressed, data_size);
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
				const size_t qsize = ccv_nnc_quantize_8i_rowwise_x(source->data.u8, datatypes[d], CCV_TENSOR_CPU_MEMORY, count, cols, formats[f], 0, 0, q->data.u8, ccv_nnc_tensor_data_size_without_padding(q->info));
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
