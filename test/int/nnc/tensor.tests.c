#include "ccv.h"
#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "3rdparty/sqlite3/sqlite3.h"
#include "3rdparty/dsfmt/dSFMT.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("tensor persistence, to / from GPU")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	sqlite3* handle;
	sqlite3_open("tensors_g.sqlite3", &handle);
	ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 20, 30), 0);
	ccv_nnc_tensor_t* const tensorG = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 20, 30), 0);
	int i;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 10 * 20 * 30; i++)
		tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(tensorG), 0);
	ccv_nnc_tensor_write(tensorG, handle, "x", 0);
	sqlite3_close(handle);
	handle = 0;
	sqlite3_open("tensors_g.sqlite3", &handle);
	ccv_nnc_tensor_t* tensor1 = 0;
	ccv_nnc_tensor_read(handle, "x", 0, 0, 0, &tensor1);
	ccv_nnc_tensor_t* tensor2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_read(handle, "x", 0, 0, 0, &tensor2);
	sqlite3_close(handle);
	ccv_nnc_tensor_t* const tensor1c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 20, 30), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor1), TENSOR_LIST(tensor1c), 0);
	REQUIRE_TENSOR_EQ(tensor1c, tensor, "the first tensor should equal to the second");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tensor2->data.f32, tensor->data.f32, 10, 1e-5, "the first 10 element should be equal");
	REQUIRE(ccv_nnc_tensor_nd(tensor2->info.dim) == 1, "should be 1-d tensor");
	REQUIRE_EQ(tensor2->info.dim[0], 10, "should be 1-d tensor with 10-element");
	ccv_nnc_tensor_free(tensor1);
	ccv_nnc_tensor_free(tensor1c);
	ccv_nnc_tensor_free(tensor2);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(tensorG);
}

TEST_CASE("tensor mapped from file")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_MPS));
	FILE* w = fopen("tensor.bin", "w+");
	float* w_a = (float*)ccmalloc(sizeof(float) * 4096 * 5);
	int i;
	for (i = 0; i < 4096 * 5; i++)
		w_a[i] = (float)(i + 1);
	fwrite(w_a, 1, sizeof(float) * 4096 * 5, w);
	fclose(w);
	ccfree(w_a);
	ccv_nnc_tensor_t* one = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	one->data.f32[0] = 1;
	ccv_nnc_tensor_t* one_gpu = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(one), TENSOR_LIST(one_gpu), 0);
	ccv_nnc_tensor_t* tensor_a = ccv_nnc_tensor_new_from_file(GPU_TENSOR_NHWC(000, 32F, 5), "tensor.bin", 0, 0);
	ccv_nnc_tensor_t* a_result = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5), 0);
	ccv_nnc_cmd_exec(CMD_ADD_FORWARD(0.5, 0.2), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor_a, one_gpu), TENSOR_LIST(a_result), 0);
	float a[] = {1 * 0.5 + 0.2, 2 * 0.5 + 0.2, 3 * 0.5 + 0.2, 4 * 0.5 + 0.2, 5 * 0.5 + 0.2};
	ccv_nnc_tensor_t* tensor_b = ccv_nnc_tensor_new_from_file(GPU_TENSOR_NHWC(000, 32F, 4), "tensor.bin", (4096 * 4 * 4), 0);
	ccv_nnc_tensor_t* b_result = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4), 0);
	ccv_nnc_cmd_exec(CMD_ADD_FORWARD(1, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor_b, one_gpu), TENSOR_LIST(b_result), 0);
	float b[] = {4096 * 4 + 1 + 1, 4096 * 4 + 2 + 1, 4096 * 4 + 3 + 1, 4096 * 4 + 4 + 1};
	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a_result, b_result), TENSOR_LIST(at, bt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, at->data.f32, a, 5, 1e-5, "the first 5 element should be equal");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, bt->data.f32, b, 4, 1e-5, "the first 4 element should be equal");
	ccv_nnc_tensor_free(tensor_a);
	ccv_nnc_tensor_free(tensor_b);
	ccv_nnc_tensor_free(one);
	ccv_nnc_tensor_free(one_gpu);
	ccv_nnc_tensor_free(a_result);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(b_result);
	ccv_nnc_tensor_free(bt);
}

static int _tensor_xor_encode(const void* const data, const size_t data_size, const int datatype, const int* const dimensions, const int dimension_count, void* const context, void* const encoded, size_t* const encoded_size, ccv_nnc_tensor_param_t* const params, unsigned int* const identifier)
{
	unsigned char* const u8 = (unsigned char*)data;
	unsigned char* const u8enc = (unsigned char*)encoded;
	int i;
	for (i = 0; i < data_size; i++)
		u8enc[i] = u8[i] ^ 0x13;
	*encoded_size = data_size;
	*identifier = 1;
	return 1;
}

static int _tensor_xor_decode(const void* const data, const size_t data_size, const int datatype, const int* const dimensions, const int dimension_count, const unsigned int identifier, void* const context, const ccv_nnc_tensor_param_t tensor_params, ccv_nnc_tensor_t** const tensor_out, void* decoded, size_t* const decoded_size)
{
	if (identifier != 1)
		return 0;
	if (!tensor_out[0])
	{
		tensor_out[0] = ccv_nnc_tensor_new(0, tensor_params, 0);
		if (!decoded)
			decoded = tensor_out[0]->data.u8;
	}
	unsigned char* const u8 = (unsigned char*)data;
	unsigned char* const u8dec = (unsigned char*)decoded;
	const size_t expected_size = *decoded_size;
	int i;
	for (i = 0; i < ccv_min(expected_size, data_size); i++)
		u8dec[i] = u8[i] ^ 0x13;
	*decoded_size = ccv_min(expected_size, data_size);
	return 1;
}

static int _tensor_noop_encode(const void* const data, const size_t data_size, const int datatype, const int* const dimensions, const int dimension_count, void* const context, void* const encoded, size_t* const encoded_size, ccv_nnc_tensor_param_t* const params, unsigned int* const identifier)
{
	return 0;
}

static int _tensor_noop_decode(const void* const data, const size_t data_size, const int datatype, const int* const dimensions, const int dimension_count, const unsigned int identifier, void* const context, const ccv_nnc_tensor_param_t tensor_params, ccv_nnc_tensor_t** const tensor_out, void* const decoded, size_t* const decoded_size)
{
	return 0;
}

TEST_CASE("tensor persistence with encoder / decoder, to / from GPU")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	sqlite3* handle;
	sqlite3_open("tensors_de_g.sqlite3", &handle);
	ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 20, 30), 0);
	int i;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 10 * 20 * 30; i++)
		tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_t* const tensorG = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 20, 30), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(tensorG), 0);
	ccv_nnc_tensor_io_option_t options = {
		.encode = _tensor_xor_encode,
		.decode = _tensor_xor_decode
	};
	ccv_nnc_tensor_write(tensorG, handle, "y", &options);
	sqlite3_close(handle);
	handle = 0;
	sqlite3_open("tensors_de_g.sqlite3", &handle);
	ccv_nnc_tensor_t* tensor1 = 0;
	ccv_nnc_tensor_read(handle, "y", &options, 0, 0, &tensor1);
	ccv_nnc_tensor_t* tensor2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_read(handle, "y", &options, 0, 0, &tensor2);
	sqlite3_close(handle);
	ccv_nnc_tensor_t* tensor1c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 20, 30), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor1), TENSOR_LIST(tensor1c), 0);
	REQUIRE_TENSOR_EQ(tensor1c, tensor, "the first tensor should equal to the second");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tensor2->data.f32, tensor->data.f32, 10, 1e-5, "the first 10 element should be equal");
	REQUIRE(ccv_nnc_tensor_nd(tensor2->info.dim) == 1, "should be 1-d tensor");
	REQUIRE_EQ(tensor2->info.dim[0], 10, "should be 1-d tensor with 10-element");
	ccv_nnc_tensor_free(tensor1);
	ccv_nnc_tensor_free(tensor1c);
	ccv_nnc_tensor_free(tensor2);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(tensorG);
}

TEST_CASE("tensor persistence with noop encoder / decoder, to / from GPU")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	sqlite3* handle;
	sqlite3_open("tensors_noop_de_g.sqlite3", &handle);
	ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 20, 30), 0);
	int i;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 10 * 20 * 30; i++)
		tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_t* const tensorG = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 20, 30), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(tensorG), 0);
	ccv_nnc_tensor_io_option_t options = {
		.encode = _tensor_noop_encode,
		.decode = _tensor_noop_decode
	};
	ccv_nnc_tensor_write(tensorG, handle, "y", &options);
	sqlite3_close(handle);
	handle = 0;
	sqlite3_open("tensors_noop_de_g.sqlite3", &handle);
	ccv_nnc_tensor_t* tensor1 = 0;
	ccv_nnc_tensor_read(handle, "y", &options, 0, 0, &tensor1);
	ccv_nnc_tensor_t* tensor2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_read(handle, "y", &options, 0, 0, &tensor2);
	sqlite3_close(handle);
	ccv_nnc_tensor_t* tensor1c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 20, 30), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor1), TENSOR_LIST(tensor1c), 0);
	REQUIRE_TENSOR_EQ(tensor1c, tensor, "the first tensor should equal to the second");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tensor2->data.f32, tensor->data.f32, 10, 1e-5, "the first 10 element should be equal");
	REQUIRE(ccv_nnc_tensor_nd(tensor2->info.dim) == 1, "should be 1-d tensor");
	REQUIRE_EQ(tensor2->info.dim[0], 10, "should be 1-d tensor with 10-element");
	ccv_nnc_tensor_free(tensor1);
	ccv_nnc_tensor_free(tensor1c);
	ccv_nnc_tensor_free(tensor2);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(tensorG);
}

TEST_CASE("tensor persistence with type coercion, to / from GPU")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	sqlite3* handle;
	sqlite3_open("tensors_tc_g.sqlite3", &handle);
	ccv_nnc_tensor_t* const tensorf32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 20, 30), 0);
	int i;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 10 * 20 * 30; i++)
		tensorf32->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_t* const tensorf16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 20, 30), 0);
	ccv_nnc_tensor_t* const tensorf16G = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 20, 30), 0);
	ccv_float_to_half_precision(tensorf32->data.f32, (uint16_t*)tensorf16->data.f16, 10 * 20 * 30);
	for (i = 0; i < 10 * 20 * 30; i++)
		tensorf32->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_t* const tensorf32G = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 20, 30), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensorf16, tensorf32), TENSOR_LIST(tensorf16G, tensorf32G), 0);
	ccv_nnc_tensor_write(tensorf16G, handle, "x", 0);
	ccv_nnc_tensor_write(tensorf32G, handle, "y", 0);
	sqlite3_close(handle);
	handle = 0;
	sqlite3_open("tensors_tc_g.sqlite3", &handle);
	ccv_nnc_tensor_t* tensor1 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* tensor1c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_read(handle, "x", 0, 0, 0, &tensor1);
	ccv_nnc_tensor_t* tensor2 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* tensor2c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_read(handle, "y", 0, 0, 0, &tensor2);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor1, tensor2), TENSOR_LIST(tensor1c, tensor2c), 0);
	sqlite3_close(handle);
	float* tensor1_ref = (float*)ccmalloc(sizeof(float) * 10);
	ccv_half_precision_to_float((uint16_t*)tensorf16->data.f16, tensor1_ref, 10);
	float* tensor2_ret = (float*)ccmalloc(sizeof(float) * 10);
	ccv_half_precision_to_float((uint16_t*)tensor2c->data.f16, tensor2_ret, 10);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tensor1c->data.f32, tensor1_ref, 10, 1e-3, "the first 10 element should be equal");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tensor2_ret, tensorf32->data.f32, 10, 1e-3, "the first 10 element should be equal");
	REQUIRE(ccv_nnc_tensor_nd(tensor1->info.dim) == 1, "should be 1-d tensor");
	REQUIRE(ccv_nnc_tensor_nd(tensor2->info.dim) == 1, "should be 1-d tensor");
	REQUIRE_EQ(tensor1->info.dim[0], 10, "should be 1-d tensor with 10-element");
	REQUIRE_EQ(tensor2->info.dim[0], 10, "should be 1-d tensor with 10-element");
	ccv_nnc_tensor_free(tensor1);
	ccv_nnc_tensor_free(tensor2);
	ccv_nnc_tensor_free(tensor1c);
	ccv_nnc_tensor_free(tensor2c);
	ccv_nnc_tensor_free(tensorf16);
	ccv_nnc_tensor_free(tensorf32);
	ccv_nnc_tensor_free(tensorf16G);
	ccv_nnc_tensor_free(tensorf32G);
	ccfree(tensor1_ref);
	ccfree(tensor2_ret);
}

TEST_CASE("tensor persistence with type coercion and encoder / decoder, to / from GPU")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	sqlite3* handle;
	sqlite3_open("tensors_tc_de_g.sqlite3", &handle);
	ccv_nnc_tensor_t* const tensorf32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 20, 30), 0);
	int i;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 10 * 20 * 30; i++)
		tensorf32->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_t* const tensorf16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 20, 30), 0);
	ccv_float_to_half_precision(tensorf32->data.f32, (uint16_t*)tensorf16->data.f16, 10 * 20 * 30);
	for (i = 0; i < 10 * 20 * 30; i++)
		tensorf32->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_t* const tensorf32G = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 20, 30), 0);
	ccv_nnc_tensor_t* const tensorf16G = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 20, 30), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensorf16, tensorf32), TENSOR_LIST(tensorf16G, tensorf32G), 0);
	ccv_nnc_tensor_io_option_t options = {
		.encode = _tensor_xor_encode,
		.decode = _tensor_xor_decode
	};
	ccv_nnc_tensor_write(tensorf16G, handle, "x", &options);
	ccv_nnc_tensor_write(tensorf32G, handle, "y", &options);
	sqlite3_close(handle);
	handle = 0;
	sqlite3_open("tensors_tc_de_g.sqlite3", &handle);
	ccv_nnc_tensor_t* tensor1 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* tensor1c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_read(handle, "x", &options, 0, 0, &tensor1);
	ccv_nnc_tensor_t* tensor2 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* tensor2c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_read(handle, "y", &options, 0, 0, &tensor2);
	sqlite3_close(handle);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor1, tensor2), TENSOR_LIST(tensor1c, tensor2c), 0);
	float* tensor1_ref = (float*)ccmalloc(sizeof(float) * 10);
	ccv_half_precision_to_float((uint16_t*)tensorf16->data.f16, tensor1_ref, 10);
	float* tensor2_ret = (float*)ccmalloc(sizeof(float) * 10);
	ccv_half_precision_to_float((uint16_t*)tensor2c->data.f16, tensor2_ret, 10);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tensor1c->data.f32, tensor1_ref, 10, 1e-3, "the first 10 element should be equal");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tensor2_ret, tensorf32->data.f32, 10, 1e-3, "the first 10 element should be equal");
	REQUIRE(ccv_nnc_tensor_nd(tensor1->info.dim) == 1, "should be 1-d tensor");
	REQUIRE(ccv_nnc_tensor_nd(tensor2->info.dim) == 1, "should be 1-d tensor");
	REQUIRE_EQ(tensor1->info.dim[0], 10, "should be 1-d tensor with 10-element");
	REQUIRE_EQ(tensor2->info.dim[0], 10, "should be 1-d tensor with 10-element");
	ccv_nnc_tensor_free(tensor1);
	ccv_nnc_tensor_free(tensor2);
	ccv_nnc_tensor_free(tensor1c);
	ccv_nnc_tensor_free(tensor2c);
	ccv_nnc_tensor_free(tensorf16);
	ccv_nnc_tensor_free(tensorf32);
	ccv_nnc_tensor_free(tensorf16G);
	ccv_nnc_tensor_free(tensorf32G);
	ccfree(tensor1_ref);
	ccfree(tensor2_ret);
}

TEST_CASE("tensor persistence with type coercion and noop encoder / decoder, to / from GPU")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	sqlite3* handle;
	sqlite3_open("tensors_tc_noop_de_g.sqlite3", &handle);
	ccv_nnc_tensor_t* const tensorf32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 20, 30), 0);
	int i;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 10 * 20 * 30; i++)
		tensorf32->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_t* const tensorf16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 20, 30), 0);
	ccv_float_to_half_precision(tensorf32->data.f32, (uint16_t*)tensorf16->data.f16, 10 * 20 * 30);
	for (i = 0; i < 10 * 20 * 30; i++)
		tensorf32->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_t* const tensorf32G = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 20, 30), 0);
	ccv_nnc_tensor_t* const tensorf16G = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 20, 30), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensorf16, tensorf32), TENSOR_LIST(tensorf16G, tensorf32G), 0);
	ccv_nnc_tensor_io_option_t options = {
		.encode = _tensor_noop_encode,
		.decode = _tensor_noop_decode
	};
	ccv_nnc_tensor_write(tensorf16G, handle, "x", &options);
	ccv_nnc_tensor_write(tensorf32G, handle, "y", &options);
	sqlite3_close(handle);
	handle = 0;
	sqlite3_open("tensors_tc_noop_de_g.sqlite3", &handle);
	ccv_nnc_tensor_t* tensor1 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* tensor1c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_read(handle, "x", &options, 0, 0, &tensor1);
	ccv_nnc_tensor_t* tensor2 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* tensor2c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_read(handle, "y", &options, 0, 0, &tensor2);
	sqlite3_close(handle);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor1, tensor2), TENSOR_LIST(tensor1c, tensor2c), 0);
	float* tensor1_ref = (float*)ccmalloc(sizeof(float) * 10);
	ccv_half_precision_to_float((uint16_t*)tensorf16->data.f16, tensor1_ref, 10);
	float* tensor2_ret = (float*)ccmalloc(sizeof(float) * 10);
	ccv_half_precision_to_float((uint16_t*)tensor2c->data.f16, tensor2_ret, 10);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tensor1c->data.f32, tensor1_ref, 10, 1e-3, "the first 10 element should be equal");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tensor2_ret, tensorf32->data.f32, 10, 1e-3, "the first 10 element should be equal");
	REQUIRE(ccv_nnc_tensor_nd(tensor1->info.dim) == 1, "should be 1-d tensor");
	REQUIRE(ccv_nnc_tensor_nd(tensor2->info.dim) == 1, "should be 1-d tensor");
	REQUIRE_EQ(tensor1->info.dim[0], 10, "should be 1-d tensor with 10-element");
	REQUIRE_EQ(tensor2->info.dim[0], 10, "should be 1-d tensor with 10-element");
	ccv_nnc_tensor_free(tensor1);
	ccv_nnc_tensor_free(tensor2);
	ccv_nnc_tensor_free(tensor1c);
	ccv_nnc_tensor_free(tensor2c);
	ccv_nnc_tensor_free(tensorf16);
	ccv_nnc_tensor_free(tensorf32);
	ccv_nnc_tensor_free(tensorf16G);
	ccv_nnc_tensor_free(tensorf32G);
	ccfree(tensor1_ref);
	ccfree(tensor2_ret);
}

#include "case_main.h"
