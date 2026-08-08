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

TEST_TEARDOWN()
{
}

TEST_CASE("scatter add a tensor")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCATTER_ADD_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_SCATTER_ADD_FORWARD, CCV_NNC_BACKEND_MPS));
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
	ccv_nnc_cmd_exec(CMD_SCATTER_ADD_FORWARD(3, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gindices), TENSOR_LIST(ga), 0);
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

TEST_CASE("scatter add a 1d tensor")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCATTER_ADD_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_SCATTER_ADD_FORWARD, CCV_NNC_BACKEND_MPS));
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
	ccv_nnc_cmd_exec(CMD_SCATTER_ADD_FORWARD(5, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gindices), TENSOR_LIST(ga), 0);
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

TEST_CASE("scatter add a tensor view")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCATTER_ADD_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_SCATTER_ADD_FORWARD, CCV_NNC_BACKEND_MPS));
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
	ccv_nnc_cmd_exec(CMD_SCATTER_ADD_FORWARD(3, 0), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)gbv, gindices), TENSOR_LIST((ccv_nnc_tensor_t*)gav), 0);
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

TEST_CASE("backward scatter add a tensor")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCATTER_ADD_BACKWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_SCATTER_ADD_BACKWARD, CCV_NNC_BACKEND_MPS));
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
	ccv_nnc_cmd_exec(CMD_SCATTER_ADD_BACKWARD(3), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, 0, gindices), TENSOR_LIST(gb), 0);
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

TEST_CASE("backward scatter add a 1d tensor")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCATTER_ADD_BACKWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_SCATTER_ADD_BACKWARD, CCV_NNC_BACKEND_MPS));
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
	ccv_nnc_cmd_exec(CMD_SCATTER_ADD_BACKWARD(5), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, 0, gindices), TENSOR_LIST(gb), 0);
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

TEST_CASE("backward scatter add a tensor view")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCATTER_ADD_BACKWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_SCATTER_ADD_BACKWARD, CCV_NNC_BACKEND_MPS));
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
	ccv_nnc_cmd_exec(CMD_SCATTER_ADD_BACKWARD(3), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)gav, 0, gindices), TENSOR_LIST((ccv_nnc_tensor_t*)gbv), 0);
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

TEST_CASE("scatter add with half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCATTER_ADD_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_SCATTER_ADD_FORWARD, CCV_NNC_BACKEND_MPS));
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
	ccv_nnc_cmd_exec(CMD_SCATTER_ADD_FORWARD(100, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gindices), TENSOR_LIST(gb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 100, 10), 0);
	ccv_nnc_cmd_exec(CMD_SCATTER_ADD_FORWARD(100, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(a16, indices), TENSOR_LIST(bt), 0);
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

TEST_CASE("backward scatter add with half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCATTER_ADD_BACKWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_SCATTER_ADD_BACKWARD, CCV_NNC_BACKEND_MPS));
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
	ccv_nnc_cmd_exec(CMD_SCATTER_ADD_BACKWARD(100), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, 0, gindices), TENSOR_LIST(gb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 10), 0);
	ccv_nnc_cmd_exec(CMD_SCATTER_ADD_BACKWARD(100), ccv_nnc_no_hint, 0, TENSOR_LIST(a16, 0, indices), TENSOR_LIST(bt), 0);
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

TEST_CASE("scatter add sums rows directly for one output with MFA")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCATTER_ADD_FORWARD, CCV_NNC_BACKEND_MPS));
	const int input_rows = 6;
	const int columns = 64;
	ccv_nnc_tensor_t* const input = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, input_rows, columns), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, input_rows), 0);
	int i;
	for (i = 0; i < input_rows * columns; i++)
		input->data.f32[i] = (float)((i % 17) - 8) * 0.125f;
	for (i = 0; i < input_rows; i++)
		indices->data.i32[i] = 0;
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, columns), 0);
	ccv_nnc_cmd_exec(CMD_SCATTER_ADD_FORWARD(1, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(input, indices), TENSOR_LIST(expected), 0);
	ccv_nnc_tensor_t* const gpu_input = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, input_rows, columns), 0);
	ccv_nnc_tensor_t* const gpu_indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, input_rows), 0);
	ccv_nnc_tensor_t* const gpu_output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, columns), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, columns), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(input, indices), TENSOR_LIST(gpu_input, gpu_indices), 0);
	ccv_nnc_cmd_exec(CMD_SCATTER_ADD_FORWARD(1, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_input, gpu_indices), TENSOR_LIST(gpu_output), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_output), TENSOR_LIST(output), 0);
	REQUIRE_TENSOR_EQ(output, expected, "single-output MFA scatter should match the CPU reference");
	ccv_nnc_tensor_free(input);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(gpu_input);
	ccv_nnc_tensor_free(gpu_indices);
	ccv_nnc_tensor_free(gpu_output);
	ccv_nnc_tensor_free(output);
}

TEST_CASE("scatter add reduces six float rows per output at 4096 outputs with MFA")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCATTER_ADD_FORWARD, CCV_NNC_BACKEND_MPS));
	const int output_rows = 4096;
	const int count_per_output = 6;
	const int input_rows = output_rows * count_per_output;
	const int columns = 64;
	ccv_nnc_tensor_t* const input = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, input_rows, columns), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, input_rows), 0);
	int i;
	for (i = 0; i < input_rows * columns; i++)
		input->data.f32[i] = (float)((i % 13) - 6) * 0.125f;
	for (i = 0; i < input_rows; i++)
		indices->data.i32[i] = (i * 5) % output_rows;
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_rows, columns), 0);
	ccv_nnc_cmd_exec(CMD_SCATTER_ADD_FORWARD(output_rows, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(input, indices), TENSOR_LIST(expected), 0);
	ccv_nnc_tensor_t* const gpu_input = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, input_rows, columns), 0);
	ccv_nnc_tensor_t* const gpu_indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, input_rows), 0);
	ccv_nnc_tensor_t* const gpu_output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_rows, columns), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_rows, columns), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(input, indices), TENSOR_LIST(gpu_input, gpu_indices), 0);
	ccv_nnc_cmd_exec(CMD_SCATTER_ADD_FORWARD(output_rows, count_per_output), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_input, gpu_indices), TENSOR_LIST(gpu_output), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_output), TENSOR_LIST(output), 0);
	REQUIRE_TENSOR_EQ(output, expected, "fixed-count MFA scatter should match the CPU reference");
	ccv_nnc_tensor_free(input);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(gpu_input);
	ccv_nnc_tensor_free(gpu_indices);
	ccv_nnc_tensor_free(gpu_output);
	ccv_nnc_tensor_free(output);
}

TEST_CASE("scatter add deterministically reduces a non-power-of-two fixed count with MFA")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCATTER_ADD_FORWARD, CCV_NNC_BACKEND_MPS));
	const int output_rows = 5;
	const int count_per_output = 17;
	const int input_rows = output_rows * count_per_output;
	const int columns = 64;
	ccv_nnc_tensor_t* const input = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, input_rows, columns), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, input_rows), 0);
	int i;
	for (i = 0; i < input_rows; i++)
		indices->data.i32[i] = i % output_rows;
	uint32_t random = 1;
	for (i = input_rows - 1; i > 0; i--)
	{
		random = random * 1664525u + 1013904223u;
		const int j = random % (i + 1);
		const int swap = indices->data.i32[i];
		indices->data.i32[i] = indices->data.i32[j];
		indices->data.i32[j] = swap;
	}
	int seen[5] = {0};
	for (i = 0; i < input_rows; i++)
	{
		const int slot = seen[indices->data.i32[i]]++;
		const float value = slot == 0 ? 1e20f : (slot == 2 ? -1e20f : 1);
		int j;
		for (j = 0; j < columns; j++)
			input->data.f32[i * columns + j] = value;
	}
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_rows, columns), 0);
	ccv_nnc_cmd_exec(CMD_SCATTER_ADD_FORWARD(output_rows, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(input, indices), TENSOR_LIST(expected), 0);
	ccv_nnc_tensor_t* const gpu_input = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, input_rows, columns), 0);
	ccv_nnc_tensor_t* const gpu_indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, input_rows), 0);
	ccv_nnc_tensor_t* const gpu_output = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_rows, columns), 0);
	ccv_nnc_tensor_t* const first_output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_rows, columns), 0);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_rows, columns), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(input, indices), TENSOR_LIST(gpu_input, gpu_indices), 0);
	ccv_nnc_cmd_exec(CMD_SCATTER_ADD_FORWARD(output_rows, count_per_output), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_input, gpu_indices), TENSOR_LIST(gpu_output), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_output), TENSOR_LIST(first_output), 0);
	REQUIRE_TENSOR_EQ(first_output, expected, "fixed-count MFA scatter should reduce in input-row order");
	for (i = 0; i < 10; i++)
	{
		ccv_nnc_cmd_exec(CMD_SCATTER_ADD_FORWARD(output_rows, count_per_output), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_input, gpu_indices), TENSOR_LIST(gpu_output), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_output), TENSOR_LIST(output), 0);
		REQUIRE_TENSOR_EQ(output, first_output, "fixed-count MFA scatter should be deterministic");
	}
	ccv_nnc_tensor_free(input);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(gpu_input);
	ccv_nnc_tensor_free(gpu_indices);
	ccv_nnc_tensor_free(gpu_output);
	ccv_nnc_tensor_free(first_output);
	ccv_nnc_tensor_free(output);
}

#include "case_main.h"
