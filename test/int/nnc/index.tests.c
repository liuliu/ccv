#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("index select a tensor")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2,
		2, 3,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	int ip[] = {1, 1};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices), TENSOR_LIST(ga, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gindices), TENSOR_LIST(gb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(b), 0);
	float btp[] = {
		2, 3,
		2, 3,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gb);
}

TEST_CASE("index select a tensor with float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2,
		2, 3,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	float ip[] = {1.5, 0.4};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices), TENSOR_LIST(ga, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gindices), TENSOR_LIST(gb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(b), 0);
	float btp[] = {
		2.5, 3.5,
		1.4, 2.4,
	};
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, btp, 4, 1e-5, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gb);
}

TEST_CASE("index select a 1d tensor")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2, 3, 4, 5
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 5), 0);
	int ip[] = {3, 2, 4};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices), TENSOR_LIST(ga, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gindices), TENSOR_LIST(gb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(b), 0);
	float btp[] = {
		4, 3, 5
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gb);
}

TEST_CASE("index select a tensor view")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2, 3, 4,
		2, 3, 4, 5,
		3, 4, 5, 6,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 3, 4), 0);
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, CPU_TENSOR_NHWC(32F, 3, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	int ip[] = {1, 1};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	memset(b->data.f32, 0, 2 * 4 * sizeof(float));
	ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, CPU_TENSOR_NHWC(32F, 2, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 4), 0);
	ccv_nnc_tensor_view_t* const gav = ccv_nnc_tensor_view_new(ga, GPU_TENSOR_NHWC(000, 32F, 3, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_view_t* const gbv = ccv_nnc_tensor_view_new(gb, GPU_TENSOR_NHWC(000, 32F, 2, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, b), TENSOR_LIST(ga, gindices, gb), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)gav, gindices), TENSOR_LIST((ccv_nnc_tensor_t*)gbv), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(b), 0);
	float btp[] = {
		0, 3, 4, 0,
		0, 3, 4, 0,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(bv);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_view_free(gav);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_view_free(gbv);
}

TEST_CASE("backward index select a tensor")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	float bp[] = {
		1, 2,
		2, 3,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	int ip[] = {1, 1};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, indices), TENSOR_LIST(gb, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, 0, gindices), TENSOR_LIST(ga), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga), TENSOR_LIST(a), 0);
	float atp[] = {
		0, 0,
		3, 5,
		0, 0,
	};
	ccv_nnc_tensor_t const at = ccv_nnc_tensor(atp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	REQUIRE_TENSOR_EQ(a, &at, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gb);
}

TEST_CASE("backward index select a 1d tensor")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	float bp[] = {
		4, 3, 5,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5), 0);
	int ip[] = {3, 2, 4};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, indices), TENSOR_LIST(gb, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, 0, gindices), TENSOR_LIST(ga), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga), TENSOR_LIST(a), 0);
	float atp[] = {
		0, 0, 3, 4, 5
	};
	ccv_nnc_tensor_t const at = ccv_nnc_tensor(atp, CPU_TENSOR_NHWC(32F, 5), 0);
	REQUIRE_TENSOR_EQ(a, &at, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gb);
}

TEST_CASE("backward index select a tensor view")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	float bp[] = {
		0, 3, 4, 0,
		0, 1, 5, 0,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 4), 0);
	int i;
	for (i = 0; i < 3 * 4; i++)
		a->data.f32[i] = i;
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, CPU_TENSOR_NHWC(32F, 3, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	int ip[] = {1, 1};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, CPU_TENSOR_NHWC(32F, 2, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 4), 0);
	ccv_nnc_tensor_view_t* const gav = ccv_nnc_tensor_view_new(ga, GPU_TENSOR_NHWC(000, 32F, 3, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_view_t* const gbv = ccv_nnc_tensor_view_new(gb, GPU_TENSOR_NHWC(000, 32F, 2, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, b), TENSOR_LIST(ga, gindices, gb), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)gbv, 0, gindices), TENSOR_LIST((ccv_nnc_tensor_t*)gav), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga), TENSOR_LIST(a), 0);
	float atp[] = {
		0, 0, 0, 3,
		4, 4, 9, 7,
		8, 0, 0, 11,
	};
	ccv_nnc_tensor_t const at = ccv_nnc_tensor(atp, CPU_TENSOR_NHWC(32F, 3, 4), 0);
	REQUIRE_TENSOR_EQ(a, &at, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(bv);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_view_free(gav);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_view_free(gbv);
}

TEST_CASE("index select forward with half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100, 10), 0);
	int i;
	for (i = 0; i < 100 * 10; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 100, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(a16), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10), 0);
	for (i = 0; i < 10; i++)
		indices->data.i32[i] = i * 9 + 1;
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 10), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 100, 10), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 10), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16, indices), TENSOR_LIST(ga, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gindices), TENSOR_LIST(gb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 10), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16, indices), TENSOR_LIST(bt), 0);
	ccv_nnc_tensor_t* const b32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 10), 0);
	ccv_nnc_tensor_t* const bt32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, bt), TENSOR_LIST(b32, bt32), 0);
	REQUIRE_TENSOR_EQ(b32, bt32, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(a16);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(b32);
	ccv_nnc_tensor_free(bt32);
}

TEST_CASE("mps index select row-wise 8i quantized tensor with float output")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 23;
	const int cols = 33;
	const int selected = 9;
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	int i;
	for (i = 0; i < rows * cols; i++)
		source->data.f32[i] = ((i * 19) % 67 - 33) / 48.0f;
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(32F, rows, cols)), 0);
	const size_t qsize = ccv_nnc_quantize_8i_rowwise(source->data.f32, CCV_32F, CCV_TENSOR_CPU_MEMORY, rows * cols, cols, 0, q->data.u8, ccv_nnc_tensor_data_size_without_padding(q->info));
	REQUIRE_EQ(qsize, ccv_nnc_tensor_data_size_without_padding(q->info), "quantized row-wise 8i size should match");
	int ip[] = {3, 5, 3, 22, 0, 11, 5, 18, 7};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, selected), 0);
	ccv_nnc_tensor_t* const dequantized = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	ccv_nnc_dequantize_8i_rowwise(q->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, qsize, cols, dequantized->data.f32, rows * cols);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, selected, cols), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dequantized, indices), TENSOR_LIST(expected), 0);
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(GPU_TENSOR_NHWC(000, 32F, rows, cols)), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, selected), 0);
	ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, selected, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q, indices), TENSOR_LIST(gq, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gq, gindices), TENSOR_LIST(gout), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, selected, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(actual), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected->data.f32, actual->data.f32, selected * cols, 1e-5, "MPS index select should dequantize only selected row-wise 8i rows");
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(gout);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(dequantized);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(source);
}

TEST_CASE("mps index select row-wise 8i quantized tensor with half output")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 17;
	const int cols = 32;
	const int selected = 8;
	float* const values = ccmalloc(sizeof(float) * rows * cols);
	int i;
	for (i = 0; i < rows * cols; i++)
		values[i] = ((i * 23) % 59 - 29) / 40.0f;
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, cols), 0);
	ccv_float_to_half_precision(values, (uint16_t*)source->data.f16, rows * cols);
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(16F, rows, cols)), 0);
	const size_t qsize = ccv_nnc_quantize_8i_rowwise(source->data.f16, CCV_16F, CCV_TENSOR_CPU_MEMORY, rows * cols, cols, 0, q->data.u8, ccv_nnc_tensor_data_size_without_padding(q->info));
	REQUIRE_EQ(qsize, ccv_nnc_tensor_data_size_without_padding(q->info), "quantized row-wise 8i size should match");
	int ip[] = {1, 12, 1, 16, 3, 0, 12, 8};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, selected), 0);
	ccv_nnc_tensor_t* const dequantized = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, cols), 0);
	ccv_nnc_dequantize_8i_rowwise(q->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, qsize, cols, dequantized->data.f16, rows * cols);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, selected, cols), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dequantized, indices), TENSOR_LIST(expected), 0);
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(GPU_TENSOR_NHWC(000, 16F, rows, cols)), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, selected), 0);
	ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, selected, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q, indices), TENSOR_LIST(gq, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gq, gindices), TENSOR_LIST(gout), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, selected, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(actual), 0);
	float* const expected_f32 = ccmalloc(sizeof(float) * selected * cols);
	float* const actual_f32 = ccmalloc(sizeof(float) * selected * cols);
	ccv_half_precision_to_float((uint16_t*)expected->data.f16, expected_f32, selected * cols);
	ccv_half_precision_to_float((uint16_t*)actual->data.f16, actual_f32, selected * cols);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_f32, actual_f32, selected * cols, 1e-3, "MPS index select should dequantize only selected half row-wise 8i rows");
	ccfree(actual_f32);
	ccfree(expected_f32);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(gout);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(dequantized);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(source);
	ccfree(values);
}

TEST_CASE("mps index select row-wise 8i quantized tensor with bfloat output")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 19;
	const int cols = 36;
	const int selected = 7;
	float* const values = ccmalloc(sizeof(float) * rows * cols);
	int i;
	for (i = 0; i < rows * cols; i++)
		values[i] = ((i * 29) % 71 - 35) / 44.0f;
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, rows, cols), 0);
	ccv_float_to_bfloat(values, (uint16_t*)source->data.f16, rows * cols);
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(16BF, rows, cols)), 0);
	const size_t qsize = ccv_nnc_quantize_8i_rowwise(source->data.f16, CCV_16BF, CCV_TENSOR_CPU_MEMORY, rows * cols, cols, 0, q->data.u8, ccv_nnc_tensor_data_size_without_padding(q->info));
	REQUIRE_EQ(qsize, ccv_nnc_tensor_data_size_without_padding(q->info), "quantized row-wise 8i size should match");
	int ip[] = {2, 18, 2, 6, 14, 0, 18};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, selected), 0);
	ccv_nnc_tensor_t* const dequantized = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, rows, cols), 0);
	ccv_nnc_dequantize_8i_rowwise(q->data.u8, CCV_16BF, CCV_TENSOR_CPU_MEMORY, qsize, cols, dequantized->data.f16, rows * cols);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, selected, cols), 0);
	for (i = 0; i < selected; i++)
		memcpy(expected->data.u8 + i * cols * sizeof(uint16_t), dequantized->data.u8 + ip[i] * cols * sizeof(uint16_t), cols * sizeof(uint16_t));
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(GPU_TENSOR_NHWC(000, 16BF, rows, cols)), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, selected), 0);
	ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, selected, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q, indices), TENSOR_LIST(gq, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gq, gindices), TENSOR_LIST(gout), 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, selected, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(actual), 0);
	float* const expected_f32 = ccmalloc(sizeof(float) * selected * cols);
	float* const actual_f32 = ccmalloc(sizeof(float) * selected * cols);
	ccv_bfloat_to_float((uint16_t*)expected->data.f16, expected_f32, selected * cols);
	ccv_bfloat_to_float((uint16_t*)actual->data.f16, actual_f32, selected * cols);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_f32, actual_f32, selected * cols, 2e-2, "MPS index select should dequantize only selected bfloat row-wise 8i rows");
	ccfree(actual_f32);
	ccfree(expected_f32);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(gout);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(dequantized);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(source);
	ccfree(values);
}

TEST_CASE("index select backward with half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 10), 0);
	int i;
	for (i = 0; i < 10 * 10; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(a16), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10), 0);
	for (i = 0; i < 10; i++)
		indices->data.i32[i] = i * 9 + 1;
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 100, 10), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 10), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 10), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 100, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16, indices), TENSOR_LIST(ga, gindices), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, 0, gindices), TENSOR_LIST(gb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 100, 10), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16, 0, indices), TENSOR_LIST(bt), 0);
	ccv_nnc_tensor_t* const b32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100, 10), 0);
	ccv_nnc_tensor_t* const bt32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, bt), TENSOR_LIST(b32, bt32), 0);
	REQUIRE_TENSOR_EQ(b32, bt32, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(a16);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(b32);
	ccv_nnc_tensor_free(bt32);
}

#include "case_main.h"
