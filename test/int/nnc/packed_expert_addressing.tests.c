#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#ifdef HAVE_MPS
#include <nnc/mps/ccv_nnc_mps.h>
#endif

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("MPS grouped IQ2_XXS SwiGLU addresses selected expert above 2^32 elements")
{
#ifdef HAVE_MPS
	GUARD_ELSE_RETURN(ccv_nnc_mfa_has_neural_accelerators(ccv_nnc_default_mfa_context()));
	const int format = CCV_NNC_QX_8I_ROWWISE_IQ2_XXS;
	const int rows = 4, n = 256, k = 256, experts = 65537;
	const size_t matrix_count = (size_t)n * k;
	// The last expert starts exactly at 2^32 decoded elements. A sparse
	// file gives it real packed weights without initializing the unused bank.
	REQUIRE_EQ((uint64_t)(experts - 1) * matrix_count, UINT64_C(1) << 32,
		"the selected expert must cross the 32-bit destination boundary");
	ccv_nnc_tensor_t* const weights = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, n, k), 0);
	for (size_t i = 0; i < matrix_count; i++)
		weights->data.f32[i] = (float)((int)((i * 37 + i / k * 13) % 127) - 63) / 512;
	const ccv_nnc_tensor_param_t packed_info = ccv_nnc_tensor_8i_rowwise_x(weights->info, format);
	ccv_nnc_tensor_t* const packed = ccv_nnc_tensor_new(0, packed_info, 0);
	const size_t packed_size = ccv_nnc_tensor_data_size_without_padding(packed_info);
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(weights->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY,
		matrix_count, k, format, 0, 0, packed->data.u8, packed_size), packed_size,
		"the selected expert must quantize successfully");
	ccv_nnc_dequantize_8i_rowwise_x(packed->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY,
		packed_size, k, format, weights->data.u8, matrix_count);
	const ccv_nnc_tensor_param_t bank_info = ccv_nnc_tensor_8i_rowwise_x(
		GPU_TENSOR_NHWC(000, 32F, experts, n, k), format);
	const size_t bank_size = ccv_nnc_tensor_data_size_without_padding(bank_info);
	const size_t payload_size = packed_size - n * sizeof(float);
	const size_t bank_scale_offset = bank_size - (size_t)experts * n * sizeof(float);
	char path[] = "/tmp/ccv-packed-expert-addressing-XXXXXX";
	const int fd = mkstemp(path);
	REQUIRE(fd >= 0, "the sparse packed-weight file should open");
	REQUIRE_EQ(ftruncate(fd, (off_t)ccv_nnc_tensor_data_size(bank_info)), 0,
		"the sparse packed bank should have the full logical size");
	REQUIRE_EQ(pwrite(fd, packed->data.u8, payload_size, (off_t)(experts - 1) * payload_size),
		(ssize_t)payload_size, "the selected expert payload should be written");
	REQUIRE_EQ(pwrite(fd, packed->data.u8 + payload_size, n * sizeof(float),
		(off_t)(bank_scale_offset + (size_t)(experts - 1) * n * sizeof(float))),
		(ssize_t)(n * sizeof(float)), "the selected expert scales should be written");
	close(fd);
	ccv_nnc_tensor_t* const bank = ccv_nnc_tensor_new_from_file(bank_info, path, 0, 0);
	unlink(path);
	REQUIRE(bank && bank->data.u8, "the packed bank should map into Metal");
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, k), 0);
	ccv_nnc_tensor_t* const hi = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
	ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, 1), 0);
	hi->data.i32[0] = experts - 1;
	hc->data.i32[0] = rows;
	for (int i = 0; i < rows * k; i++)
		ha->data.f32[i] = (float)((i * 17 + i / k * 11) % 255 - 127) / 128;
	for (int i = 0; i < rows; i++)
		hr->data.f32[i] = (float)(i + 1) / 8;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, k), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 1), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 1), 0);
	ccv_nnc_tensor_t* const route = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, 1), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, n), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha, hi, hc, hr), TENSOR_LIST(a, indices, counts, route), stream);
	ccv_nnc_cmd_t command = CMD_SEGMENTED_SWIGLU_FORWARD(10);
	command.backend = CCV_NNC_BACKEND_MPS;
	const int status = ccv_nnc_cmd_exec(command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(a, indices, counts, bank, bank, route), TENSOR_LIST(output), stream);
	REQUIRE_EQ(status, CCV_NNC_EXEC_SUCCESS, "the grouped packed expert should execute above 2^32 elements");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(output), TENSOR_LIST(actual), stream);
	ccv_nnc_stream_context_wait(stream);
	double squared_error = 0, squared_reference = 0, max_error = 0;
	for (int row = 0; row < rows; row++)
		for (int col = 0; col < n; col++)
		{
			float sum = 0;
			for (int inner = 0; inner < k; inner++)
				sum += ha->data.f32[row * k + inner] * weights->data.f32[col * k + inner];
			const float gate = ccv_min(sum, 10);
			const float up = ccv_min(ccv_max(sum, -10), 10);
			const float expected = hr->data.f32[row] * gate * up / (1 + expf(-gate));
			const float value = actual->data.f32[row * n + col];
			REQUIRE(isfinite(value), "the selected expert output should be finite");
			const double error = value - expected;
			squared_error += error * error;
			squared_reference += (double)expected * expected;
			max_error = ccv_max(max_error, fabs(error));
		}
	REQUIRE(squared_reference > 1e-3, "the selected expert must produce a nonzero reference");
	REQUIRE(sqrt(squared_error / squared_reference) < 0.02 && max_error < 0.04,
		"large-bank output should match the CPU decoder/reference: relL2=%g max=%g",
		sqrt(squared_error / squared_reference), max_error);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(route);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(hr);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hi);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(bank);
	ccv_nnc_tensor_free(packed);
	ccv_nnc_tensor_free(weights);
#else
	GUARD_ELSE_RETURN(0);
#endif
}

TEST_CASE("MPS grouped Q2_K GEMM addresses selected expert above 2^32 elements")
{
#ifdef HAVE_MPS
	GUARD_ELSE_RETURN(ccv_nnc_mfa_has_neural_accelerators(ccv_nnc_default_mfa_context()));
	const int format = CCV_NNC_QX_8I_ROWWISE_Q2_K;
	const int rows = 4, n = 256, k = 256, experts = 65537;
	const size_t matrix_count = (size_t)n * k;
	// The last expert starts exactly at 2^32 decoded elements. A sparse
	// file gives it real packed weights without initializing the unused bank.
	REQUIRE_EQ((uint64_t)(experts - 1) * matrix_count, UINT64_C(1) << 32,
		"the selected expert must cross the 32-bit destination boundary");
	ccv_nnc_tensor_t* const weights = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, n, k), 0);
	for (size_t i = 0; i < matrix_count; i++)
		weights->data.f32[i] = (float)((int)((i * 37 + i / k * 13) % 127) - 63) / 512;
	const ccv_nnc_tensor_param_t packed_info = ccv_nnc_tensor_8i_rowwise_x(weights->info, format);
	ccv_nnc_tensor_t* const packed = ccv_nnc_tensor_new(0, packed_info, 0);
	const size_t packed_size = ccv_nnc_tensor_data_size_without_padding(packed_info);
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(weights->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY,
		matrix_count, k, format, 0, 0, packed->data.u8, packed_size), packed_size,
		"the selected expert must quantize successfully");
	ccv_nnc_dequantize_8i_rowwise_x(packed->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY,
		packed_size, k, format, weights->data.u8, matrix_count);
	const ccv_nnc_tensor_param_t bank_info = ccv_nnc_tensor_8i_rowwise_x(
		GPU_TENSOR_NHWC(000, 32F, experts, n, k), format);
	const size_t bank_size = ccv_nnc_tensor_data_size_without_padding(bank_info);
	const size_t payload_size = packed_size - n * sizeof(float);
	const size_t bank_scale_offset = bank_size - (size_t)experts * n * sizeof(float);
	char path[] = "/tmp/ccv-packed-expert-addressing-XXXXXX";
	const int fd = mkstemp(path);
	REQUIRE(fd >= 0, "the sparse packed-weight file should open");
	REQUIRE_EQ(ftruncate(fd, (off_t)ccv_nnc_tensor_data_size(bank_info)), 0,
		"the sparse packed bank should have the full logical size");
	REQUIRE_EQ(pwrite(fd, packed->data.u8, payload_size, (off_t)(experts - 1) * payload_size),
		(ssize_t)payload_size, "the selected expert payload should be written");
	REQUIRE_EQ(pwrite(fd, packed->data.u8 + payload_size, n * sizeof(float),
		(off_t)(bank_scale_offset + (size_t)(experts - 1) * n * sizeof(float))),
		(ssize_t)(n * sizeof(float)), "the selected expert scales should be written");
	close(fd);
	ccv_nnc_tensor_t* const bank = ccv_nnc_tensor_new_from_file(bank_info, path, 0, 0);
	unlink(path);
	REQUIRE(bank && bank->data.u8, "the packed bank should map into Metal");
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, k), 0);
	ccv_nnc_tensor_t* const hi = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
	hi->data.i32[0] = experts - 1;
	hc->data.i32[0] = rows;
	for (int i = 0; i < rows * k; i++)
		ha->data.f32[i] = (float)((i * 17 + i / k * 11) % 255 - 127) / 128;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, k), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 1), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 1), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, n), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha, hi, hc), TENSOR_LIST(a, indices, counts), stream);
	ccv_nnc_cmd_t command = CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2));
	command.backend = CCV_NNC_BACKEND_MPS;
	const int status = ccv_nnc_cmd_exec(command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(a, indices, counts, bank), TENSOR_LIST(output), stream);
	REQUIRE_EQ(status, CCV_NNC_EXEC_SUCCESS, "the grouped packed expert should execute above 2^32 elements");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(output), TENSOR_LIST(actual), stream);
	ccv_nnc_stream_context_wait(stream);
	double squared_error = 0, squared_reference = 0, max_error = 0;
	for (int row = 0; row < rows; row++)
		for (int col = 0; col < n; col++)
		{
			float sum = 0;
			for (int inner = 0; inner < k; inner++)
				sum += ha->data.f32[row * k + inner] * weights->data.f32[col * k + inner];
			const float expected = sum;
			const float value = actual->data.f32[row * n + col];
			REQUIRE(isfinite(value), "the selected expert output should be finite");
			const double error = value - expected;
			squared_error += error * error;
			squared_reference += (double)expected * expected;
			max_error = ccv_max(max_error, fabs(error));
		}
	REQUIRE(squared_reference > 1e-3, "the selected expert must produce a nonzero reference");
	REQUIRE(sqrt(squared_error / squared_reference) < 0.02 && max_error < 0.04,
		"large-bank output should match the CPU decoder/reference: relL2=%g max=%g",
		sqrt(squared_error / squared_reference), max_error);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hi);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(bank);
	ccv_nnc_tensor_free(packed);
	ccv_nnc_tensor_free(weights);
#else
	GUARD_ELSE_RETURN(0);
#endif
}

// This full-expansion fallback case requires more than 16 GiB of GPU memory.
TEST_CASE("MPS generic IQ2_XXS SwiGLU addresses selected expert above 2^32 elements")
{
#ifdef HAVE_MPS
	GUARD_ELSE_RETURN(ccv_nnc_mfa_context_supported(ccv_nnc_default_mfa_context()));
	const int format = CCV_NNC_QX_8I_ROWWISE_IQ2_XXS;
	const int rows = 4, n = 256, k = 256, experts = 65537;
	const size_t matrix_count = (size_t)n * k;
	// The last expert starts exactly at 2^32 decoded elements. A sparse
	// file gives it real packed weights without initializing the unused bank.
	REQUIRE_EQ((uint64_t)(experts - 1) * matrix_count, UINT64_C(1) << 32,
		"the selected expert must cross the 32-bit destination boundary");
	ccv_nnc_tensor_t* const weights = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, n, k), 0);
	for (size_t i = 0; i < matrix_count; i++)
		weights->data.f32[i] = (float)((int)((i * 37 + i / k * 13) % 127) - 63) / 512;
	const ccv_nnc_tensor_param_t packed_info = ccv_nnc_tensor_8i_rowwise_x(weights->info, format);
	ccv_nnc_tensor_t* const packed = ccv_nnc_tensor_new(0, packed_info, 0);
	const size_t packed_size = ccv_nnc_tensor_data_size_without_padding(packed_info);
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(weights->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY,
		matrix_count, k, format, 0, 0, packed->data.u8, packed_size), packed_size,
		"the selected expert must quantize successfully");
	ccv_nnc_dequantize_8i_rowwise_x(packed->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY,
		packed_size, k, format, weights->data.u8, matrix_count);
	const ccv_nnc_tensor_param_t bank_info = ccv_nnc_tensor_8i_rowwise_x(
		GPU_TENSOR_NHWC(000, 32F, experts, n, k), format);
	const size_t bank_size = ccv_nnc_tensor_data_size_without_padding(bank_info);
	const size_t payload_size = packed_size - n * sizeof(float);
	const size_t bank_scale_offset = bank_size - (size_t)experts * n * sizeof(float);
	char path[] = "/tmp/ccv-packed-expert-addressing-XXXXXX";
	const int fd = mkstemp(path);
	REQUIRE(fd >= 0, "the sparse packed-weight file should open");
	REQUIRE_EQ(ftruncate(fd, (off_t)ccv_nnc_tensor_data_size(bank_info)), 0,
		"the sparse packed bank should have the full logical size");
	REQUIRE_EQ(pwrite(fd, packed->data.u8, payload_size, (off_t)(experts - 1) * payload_size),
		(ssize_t)payload_size, "the selected expert payload should be written");
	REQUIRE_EQ(pwrite(fd, packed->data.u8 + payload_size, n * sizeof(float),
		(off_t)(bank_scale_offset + (size_t)(experts - 1) * n * sizeof(float))),
		(ssize_t)(n * sizeof(float)), "the selected expert scales should be written");
	close(fd);
	ccv_nnc_tensor_t* const bank = ccv_nnc_tensor_new_from_file(bank_info, path, 0, 0);
	unlink(path);
	REQUIRE(bank && bank->data.u8, "the packed bank should map into Metal");
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, k), 0);
	ccv_nnc_tensor_t* const hi = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
	ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, 1), 0);
	hi->data.i32[0] = experts - 1;
	hc->data.i32[0] = rows;
	for (int i = 0; i < rows * k; i++)
		ha->data.f32[i] = (float)((i * 17 + i / k * 11) % 255 - 127) / 128;
	for (int i = 0; i < rows; i++)
		hr->data.f32[i] = (float)(i + 1) / 8;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, k), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 1), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 1), 0);
	ccv_nnc_tensor_t* const route = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, 1), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, n), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha, hi, hc, hr), TENSOR_LIST(a, indices, counts, route), stream);
	ccv_nnc_cmd_t command = CMD_SEGMENTED_SWIGLU_FORWARD(10);
	command.backend = CCV_NNC_BACKEND_MPS;
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	const int status = ccv_nnc_cmd_exec(command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(a, indices, counts, bank, bank, route), TENSOR_LIST(output), stream);
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	REQUIRE_EQ(status, CCV_NNC_EXEC_SUCCESS, "the grouped packed expert should execute above 2^32 elements");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(output), TENSOR_LIST(actual), stream);
	ccv_nnc_stream_context_wait(stream);
	double squared_error = 0, squared_reference = 0, max_error = 0;
	for (int row = 0; row < rows; row++)
		for (int col = 0; col < n; col++)
		{
			float sum = 0;
			for (int inner = 0; inner < k; inner++)
				sum += ha->data.f32[row * k + inner] * weights->data.f32[col * k + inner];
			const float gate = ccv_min(sum, 10);
			const float up = ccv_min(ccv_max(sum, -10), 10);
			const float expected = hr->data.f32[row] * gate * up / (1 + expf(-gate));
			const float value = actual->data.f32[row * n + col];
			REQUIRE(isfinite(value), "the selected expert output should be finite");
			const double error = value - expected;
			squared_error += error * error;
			squared_reference += (double)expected * expected;
			max_error = ccv_max(max_error, fabs(error));
		}
	REQUIRE(squared_reference > 1e-3, "the selected expert must produce a nonzero reference");
	REQUIRE(sqrt(squared_error / squared_reference) < 0.02 && max_error < 0.04,
		"large-bank output should match the CPU decoder/reference: relL2=%g max=%g",
		sqrt(squared_error / squared_reference), max_error);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(route);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(hr);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hi);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(bank);
	ccv_nnc_tensor_free(packed);
	ccv_nnc_tensor_free(weights);
#else
	GUARD_ELSE_RETURN(0);
#endif
}

// This full-expansion fallback case requires more than 16 GiB of GPU memory.
TEST_CASE("MPS generic Q2_K GEMM addresses selected expert above 2^32 elements")
{
#ifdef HAVE_MPS
	GUARD_ELSE_RETURN(ccv_nnc_mfa_context_supported(ccv_nnc_default_mfa_context()));
	const int format = CCV_NNC_QX_8I_ROWWISE_Q2_K;
	const int rows = 4, n = 256, k = 256, experts = 65537;
	const size_t matrix_count = (size_t)n * k;
	// The last expert starts exactly at 2^32 decoded elements. A sparse
	// file gives it real packed weights without initializing the unused bank.
	REQUIRE_EQ((uint64_t)(experts - 1) * matrix_count, UINT64_C(1) << 32,
		"the selected expert must cross the 32-bit destination boundary");
	ccv_nnc_tensor_t* const weights = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, n, k), 0);
	for (size_t i = 0; i < matrix_count; i++)
		weights->data.f32[i] = (float)((int)((i * 37 + i / k * 13) % 127) - 63) / 512;
	const ccv_nnc_tensor_param_t packed_info = ccv_nnc_tensor_8i_rowwise_x(weights->info, format);
	ccv_nnc_tensor_t* const packed = ccv_nnc_tensor_new(0, packed_info, 0);
	const size_t packed_size = ccv_nnc_tensor_data_size_without_padding(packed_info);
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(weights->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY,
		matrix_count, k, format, 0, 0, packed->data.u8, packed_size), packed_size,
		"the selected expert must quantize successfully");
	ccv_nnc_dequantize_8i_rowwise_x(packed->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY,
		packed_size, k, format, weights->data.u8, matrix_count);
	const ccv_nnc_tensor_param_t bank_info = ccv_nnc_tensor_8i_rowwise_x(
		GPU_TENSOR_NHWC(000, 32F, experts, n, k), format);
	const size_t bank_size = ccv_nnc_tensor_data_size_without_padding(bank_info);
	const size_t payload_size = packed_size - n * sizeof(float);
	const size_t bank_scale_offset = bank_size - (size_t)experts * n * sizeof(float);
	char path[] = "/tmp/ccv-packed-expert-addressing-XXXXXX";
	const int fd = mkstemp(path);
	REQUIRE(fd >= 0, "the sparse packed-weight file should open");
	REQUIRE_EQ(ftruncate(fd, (off_t)ccv_nnc_tensor_data_size(bank_info)), 0,
		"the sparse packed bank should have the full logical size");
	REQUIRE_EQ(pwrite(fd, packed->data.u8, payload_size, (off_t)(experts - 1) * payload_size),
		(ssize_t)payload_size, "the selected expert payload should be written");
	REQUIRE_EQ(pwrite(fd, packed->data.u8 + payload_size, n * sizeof(float),
		(off_t)(bank_scale_offset + (size_t)(experts - 1) * n * sizeof(float))),
		(ssize_t)(n * sizeof(float)), "the selected expert scales should be written");
	close(fd);
	ccv_nnc_tensor_t* const bank = ccv_nnc_tensor_new_from_file(bank_info, path, 0, 0);
	unlink(path);
	REQUIRE(bank && bank->data.u8, "the packed bank should map into Metal");
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, k), 0);
	ccv_nnc_tensor_t* const hi = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
	hi->data.i32[0] = experts - 1;
	hc->data.i32[0] = rows;
	for (int i = 0; i < rows * k; i++)
		ha->data.f32[i] = (float)((i * 17 + i / k * 11) % 255 - 127) / 128;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, k), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 1), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 1), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, n), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha, hi, hc), TENSOR_LIST(a, indices, counts), stream);
	ccv_nnc_cmd_t command = CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2));
	command.backend = CCV_NNC_BACKEND_MPS;
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	const int status = ccv_nnc_cmd_exec(command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(a, indices, counts, bank), TENSOR_LIST(output), stream);
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	REQUIRE_EQ(status, CCV_NNC_EXEC_SUCCESS, "the grouped packed expert should execute above 2^32 elements");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(output), TENSOR_LIST(actual), stream);
	ccv_nnc_stream_context_wait(stream);
	double squared_error = 0, squared_reference = 0, max_error = 0;
	for (int row = 0; row < rows; row++)
		for (int col = 0; col < n; col++)
		{
			float sum = 0;
			for (int inner = 0; inner < k; inner++)
				sum += ha->data.f32[row * k + inner] * weights->data.f32[col * k + inner];
			const float expected = sum;
			const float value = actual->data.f32[row * n + col];
			REQUIRE(isfinite(value), "the selected expert output should be finite");
			const double error = value - expected;
			squared_error += error * error;
			squared_reference += (double)expected * expected;
			max_error = ccv_max(max_error, fabs(error));
		}
	REQUIRE(squared_reference > 1e-3, "the selected expert must produce a nonzero reference");
	REQUIRE(sqrt(squared_error / squared_reference) < 0.02 && max_error < 0.04,
		"large-bank output should match the CPU decoder/reference: relL2=%g max=%g",
		sqrt(squared_error / squared_reference), max_error);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hi);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(bank);
	ccv_nnc_tensor_free(packed);
	ccv_nnc_tensor_free(weights);
#else
	GUARD_ELSE_RETURN(0);
#endif
}

// Match V4.1's projection dimensions and a 4096-token, top-6 prefill. Repeated
// quantized rows with distinct FP32 scales give a cheap independent CPU reference.
// The reference includes per-row INT8 activation rounding used by the NA path.
TEST_CASE("MPS grouped IQ2_XXS SwiGLU handles V4.1 dimensions and 4096 tokens")
{
#ifdef HAVE_MPS
	GUARD_ELSE_RETURN(ccv_nnc_mfa_has_neural_accelerators(ccv_nnc_default_mfa_context()));
	const int rows = 4096 * 6, n = 2304, k = 5120;
	const int experts = 384, active = 8;
	const int selected[8] = { 0, 31, 63, 127, 191, 255, 319, 383 };
	const int format = CCV_NNC_QX_8I_ROWWISE_IQ2_XXS;
	ccv_nnc_tensor_t* const weight_row = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, k), 0);
	uint32_t random = 19;
	for (int i = 0; i < k; i++)
	{
		random = random * 1664525u + 1013904223u;
		weight_row->data.f32[i] = ((float)(random >> 16) / 65536 - 0.5f) / 8;
	}
	const ccv_nnc_tensor_param_t row_info = ccv_nnc_tensor_8i_rowwise_x(weight_row->info, format);
	ccv_nnc_tensor_t* const packed_row = ccv_nnc_tensor_new(0, row_info, 0);
	const size_t row_size = ccv_nnc_tensor_data_size_without_padding(row_info);
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(weight_row->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY,
		k, k, format, 0, 0, packed_row->data.u8, row_size), row_size, "weight row should quantize");
	ccv_nnc_dequantize_8i_rowwise_x(packed_row->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY,
		row_size, k, format, weight_row->data.u8, k);
	const size_t row_payload = (size_t)k / ccv_nnc_8i_rowwise_x_group_size(format) * ccv_nnc_8i_rowwise_x_group_bits(format) / 8;
	float row_scale;
	memcpy(&row_scale, packed_row->data.u8 + ((row_payload + 127) & ~(size_t)127), sizeof(float));
	const ccv_nnc_tensor_param_t bank_info = ccv_nnc_tensor_8i_rowwise_x(GPU_TENSOR_NHWC(000, 32F, experts, n, k), format);
	const size_t bank_size = ccv_nnc_tensor_data_size_without_padding(bank_info);
	const size_t scale_offset = bank_size - (size_t)experts * n * sizeof(float);
	unsigned char* const payload = ccmalloc(row_payload * n);
	float* const scales = ccmalloc(n * sizeof(float));
	for (int i = 0; i < n; i++)
	{
		memcpy(payload + i * row_payload, packed_row->data.u8, row_payload);
		scales[i] = row_scale * (0.5f + (float)(i % 127) / 127);
	}
	char path[] = "/tmp/ccv-packed-prefill-XXXXXX";
	const int fd = mkstemp(path);
	REQUIRE(fd >= 0, "the sparse prefill bank should open");
	REQUIRE_EQ(ftruncate(fd, (off_t)ccv_nnc_tensor_data_size(bank_info)), 0, "bank should have full logical size");
	for (int i = 0; i < active; i++)
	{
		REQUIRE_EQ(pwrite(fd, payload, row_payload * n, (off_t)selected[i] * row_payload * n),
			(ssize_t)(row_payload * n), "selected expert payload should be written");
		REQUIRE_EQ(pwrite(fd, scales, n * sizeof(float), (off_t)scale_offset + (off_t)selected[i] * n * sizeof(float)),
			(ssize_t)(n * sizeof(float)), "selected expert scales should be written");
	}
	close(fd);
	ccfree(scales);
	ccfree(payload);
	ccv_nnc_tensor_t* const bank = ccv_nnc_tensor_new_from_file(bank_info, path, 0, 0);
	unlink(path);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, k), 0);
	ccv_nnc_tensor_t* const hi = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, experts), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, experts), 0);
	ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, 1), 0);
	for (int i = 0; i < rows * k; i++)
	{
		random = random * 1664525u + 1013904223u;
		ha->data.f32[i] = (float)(random >> 16) / 65536 - 0.5f;
	}
	for (int i = 0; i < rows; i++)
		hr->data.f32[i] = (float)(i % 6 + 1) / 7;
	for (int i = 0; i < experts; i++)
	{
		hi->data.i32[i] = i < active ? selected[i] : 0;
		hc->data.i32[i] = i < active ? rows / active : 0;
	}
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, k), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, experts), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, experts), 0);
	ccv_nnc_tensor_t* const route = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, 1), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, n), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha, hi, hc, hr), TENSOR_LIST(a, indices, counts, route), stream);
	ccv_nnc_cmd_t command = CMD_SEGMENTED_SWIGLU_FORWARD(10);
	command.backend = CCV_NNC_BACKEND_MPS;
	const int status = ccv_nnc_cmd_exec(command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(a, indices, counts, bank, bank, route), TENSOR_LIST(output), stream);
	REQUIRE_EQ(status, CCV_NNC_EXEC_SUCCESS, "4096-token packed prefill should execute");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(output), TENSOR_LIST(actual), stream);
	ccv_nnc_stream_context_wait(stream);
	double squared_error = 0, squared_reference = 0, max_error = 0;
	for (int row = 0; row < rows; row++)
	{
		float maximum = 0;
		for (int inner = 0; inner < k; inner++)
			maximum = ccv_max(maximum, fabsf(ha->data.f32[row * k + inner]));
		const float activation_scale = maximum / 127;
		const float inverse_scale = 127 / maximum;
		int32_t dot = 0;
		for (int inner = 0; inner < k; inner++)
			dot += (int32_t)rintf(ha->data.f32[row * k + inner] * inverse_scale) *
				(int32_t)rintf(weight_row->data.f32[inner] / row_scale);
		const float sum = (float)dot * activation_scale;
		for (int col = 0; col < n; col++)
		{
			const float projected = sum * (row_scale * (0.5f + (float)(col % 127) / 127));
			const float gate = ccv_min(projected, 10), up = ccv_min(ccv_max(projected, -10), 10);
			const float expected = hr->data.f32[row] * gate * up / (1 + expf(-gate));
			const float value = actual->data.f32[row * n + col];
			REQUIRE(isfinite(value), "4096-token output should be finite at row %d col %d", row, col);
			const double error = value - expected;
			squared_error += error * error;
			squared_reference += (double)expected * expected;
			max_error = ccv_max(max_error, fabs(error) / ccv_max(fabs(expected), 1));
		}
	}
	REQUIRE(squared_reference > 1, "reference should be nonzero");
	REQUIRE(sqrt(squared_error / squared_reference) < 1e-4 && max_error < 1e-3,
		"4096-token prefill should match CPU reference: relL2=%g maxNormalized=%g", sqrt(squared_error / squared_reference), max_error);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(route);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(hr);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hi);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(bank);
	ccv_nnc_tensor_free(packed_row);
	ccv_nnc_tensor_free(weight_row);
#else
	GUARD_ELSE_RETURN(0);
#endif
}

// Match V4.1's projection dimensions and a 4096-token, top-6 prefill. Repeated
// quantized rows with distinct FP32 scales give a cheap independent CPU reference.
// The reference includes per-row INT8 activation rounding used by the NA path.
TEST_CASE("MPS grouped Q2_K GEMM handles V4.1 dimensions and 4096 tokens")
{
#ifdef HAVE_MPS
	GUARD_ELSE_RETURN(ccv_nnc_mfa_has_neural_accelerators(ccv_nnc_default_mfa_context()));
	const int rows = 4096 * 6, n = 5120, k = 2304;
	const int experts = 384, active = 8;
	const int selected[8] = { 0, 31, 63, 127, 191, 255, 319, 383 };
	const int format = CCV_NNC_QX_8I_ROWWISE_Q2_K;
	ccv_nnc_tensor_t* const weight_row = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, k), 0);
	uint32_t random = 19;
	for (int i = 0; i < k; i++)
	{
		random = random * 1664525u + 1013904223u;
		weight_row->data.f32[i] = ((float)(random >> 16) / 65536 - 0.5f) / 8;
	}
	const ccv_nnc_tensor_param_t row_info = ccv_nnc_tensor_8i_rowwise_x(weight_row->info, format);
	ccv_nnc_tensor_t* const packed_row = ccv_nnc_tensor_new(0, row_info, 0);
	const size_t row_size = ccv_nnc_tensor_data_size_without_padding(row_info);
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(weight_row->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY,
		k, k, format, 0, 0, packed_row->data.u8, row_size), row_size, "weight row should quantize");
	ccv_nnc_dequantize_8i_rowwise_x(packed_row->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY,
		row_size, k, format, weight_row->data.u8, k);
	const size_t row_payload = (size_t)k / ccv_nnc_8i_rowwise_x_group_size(format) * ccv_nnc_8i_rowwise_x_group_bits(format) / 8;
	float row_scale;
	memcpy(&row_scale, packed_row->data.u8 + ((row_payload + 127) & ~(size_t)127), sizeof(float));
	const ccv_nnc_tensor_param_t bank_info = ccv_nnc_tensor_8i_rowwise_x(GPU_TENSOR_NHWC(000, 32F, experts, n, k), format);
	const size_t bank_size = ccv_nnc_tensor_data_size_without_padding(bank_info);
	const size_t scale_offset = bank_size - (size_t)experts * n * sizeof(float);
	unsigned char* const payload = ccmalloc(row_payload * n);
	float* const scales = ccmalloc(n * sizeof(float));
	for (int i = 0; i < n; i++)
	{
		memcpy(payload + i * row_payload, packed_row->data.u8, row_payload);
		scales[i] = row_scale * (0.5f + (float)(i % 127) / 127);
	}
	char path[] = "/tmp/ccv-packed-prefill-XXXXXX";
	const int fd = mkstemp(path);
	REQUIRE(fd >= 0, "the sparse prefill bank should open");
	REQUIRE_EQ(ftruncate(fd, (off_t)ccv_nnc_tensor_data_size(bank_info)), 0, "bank should have full logical size");
	for (int i = 0; i < active; i++)
	{
		REQUIRE_EQ(pwrite(fd, payload, row_payload * n, (off_t)selected[i] * row_payload * n),
			(ssize_t)(row_payload * n), "selected expert payload should be written");
		REQUIRE_EQ(pwrite(fd, scales, n * sizeof(float), (off_t)scale_offset + (off_t)selected[i] * n * sizeof(float)),
			(ssize_t)(n * sizeof(float)), "selected expert scales should be written");
	}
	close(fd);
	ccfree(scales);
	ccfree(payload);
	ccv_nnc_tensor_t* const bank = ccv_nnc_tensor_new_from_file(bank_info, path, 0, 0);
	unlink(path);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, k), 0);
	ccv_nnc_tensor_t* const hi = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, experts), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, experts), 0);
	for (int i = 0; i < rows * k; i++)
	{
		random = random * 1664525u + 1013904223u;
		ha->data.f32[i] = (float)(random >> 16) / 65536 - 0.5f;
	}
	for (int i = 0; i < experts; i++)
	{
		hi->data.i32[i] = i < active ? selected[i] : 0;
		hc->data.i32[i] = i < active ? rows / active : 0;
	}
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, k), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, experts), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, experts), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, n), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha, hi, hc), TENSOR_LIST(a, indices, counts), stream);
	ccv_nnc_cmd_t command = CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2));
	command.backend = CCV_NNC_BACKEND_MPS;
	const int status = ccv_nnc_cmd_exec(command, ccv_nnc_no_hint, 0,
		TENSOR_LIST(a, indices, counts, bank), TENSOR_LIST(output), stream);
	REQUIRE_EQ(status, CCV_NNC_EXEC_SUCCESS, "4096-token packed prefill should execute");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(output), TENSOR_LIST(actual), stream);
	ccv_nnc_stream_context_wait(stream);
	double squared_error = 0, squared_reference = 0, max_error = 0;
	for (int row = 0; row < rows; row++)
	{
		float maximum = 0;
		for (int inner = 0; inner < k; inner++)
			maximum = ccv_max(maximum, fabsf(ha->data.f32[row * k + inner]));
		const float activation_scale = maximum / 127;
		const float inverse_scale = 127 / maximum;
		int32_t dot = 0;
		for (int inner = 0; inner < k; inner++)
			dot += (int32_t)rintf(ha->data.f32[row * k + inner] * inverse_scale) *
				(int32_t)rintf(weight_row->data.f32[inner] / row_scale);
		const float sum = (float)dot * activation_scale;
		for (int col = 0; col < n; col++)
		{
			const float projected = sum * (row_scale * (0.5f + (float)(col % 127) / 127));
			const float expected = projected;
			const float value = actual->data.f32[row * n + col];
			REQUIRE(isfinite(value), "4096-token output should be finite at row %d col %d", row, col);
			const double error = value - expected;
			squared_error += error * error;
			squared_reference += (double)expected * expected;
			max_error = ccv_max(max_error, fabs(error) / ccv_max(fabs(expected), 1));
		}
	}
	REQUIRE(squared_reference > 1, "reference should be nonzero");
	REQUIRE(sqrt(squared_error / squared_reference) < 1e-4 && max_error < 1e-3,
		"4096-token prefill should match CPU reference: relL2=%g maxNormalized=%g", sqrt(squared_error / squared_reference), max_error);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hi);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(bank);
	ccv_nnc_tensor_free(packed_row);
	ccv_nnc_tensor_free(weight_row);
#else
	GUARD_ELSE_RETURN(0);
#endif
}

#include "case_main.h"
