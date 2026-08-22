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

TEST_CASE("reduce sum forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_SUM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_SUM_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_SUM_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_SUM_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("reduce sum forward noop")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_SUM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_SUM_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_SUM_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_SUM_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("reduce sum backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_SUM_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	hb->data.f32[0] = 1;
	hb->data.f32[1] = 2;
	hb->data.f32[2] = 3;
	ccv_nnc_cmd_exec(CMD_REDUCE_SUM_BACKWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(hb), TENSOR_LIST(ha), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hb), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_SUM_BACKWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(a), 0);
	ccv_nnc_tensor_t* const at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(at), 0);
	REQUIRE_TENSOR_EQ(ha, at, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(at);
}

TEST_CASE("reduce mean forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_MEAN_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_MEAN_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_MEAN_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_MEAN_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("reduce mean forward noop")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_MEAN_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_MEAN_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_MEAN_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_MEAN_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("reduce mean backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_MEAN_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_MEAN_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	hb->data.f32[0] = 1;
	hb->data.f32[1] = 2;
	hb->data.f32[2] = 3;
	ccv_nnc_cmd_exec(CMD_REDUCE_MEAN_BACKWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(hb), TENSOR_LIST(ha), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hb), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_MEAN_BACKWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(a), 0);
	ccv_nnc_tensor_t* const at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(at), 0);
	REQUIRE_TENSOR_EQ(ha, at, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(at);
}

TEST_CASE("reduce logsumexp forward and backward on MPS")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_LOGSUMEXP_FORWARD, CCV_NNC_BACKEND_MPS) && ccv_nnc_cmd_ok(CCV_NNC_REDUCE_LOGSUMEXP_BACKWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 37;
	const int columns = 257;
	const float scale = 0.625f;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, columns), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, 1), 0);
	ccv_nnc_tensor_t* const hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, 1), 0);
	ccv_nnc_tensor_t* const hh = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, columns), 0);
	int i;
	for (i = 0; i < rows * columns; i++)
		ha->data.f32[i] = 40 * dsfmt_genrand_open_close(&dsfmt) - 20;
	for (i = 0; i < rows; i++)
		hg->data.f32[i] = 2 * dsfmt_genrand_open_close(&dsfmt) - 1;
	ccv_nnc_cmd_exec(CMD_REDUCE_LOGSUMEXP_FORWARD(scale, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_LOGSUMEXP_BACKWARD(scale, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hb), TENSOR_LIST(hh), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, columns), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, 1), 0);
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, 1), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, columns), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_LOGSUMEXP_FORWARD(scale, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_LOGSUMEXP_BACKWARD(scale, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(h), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, 1), 0);
	ccv_nnc_tensor_t* const ht = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, columns), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, h), TENSOR_LIST(bt, ht), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hb->data.f32, bt->data.f32, rows, 1e-5, "MPS logsumexp should match CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hh->data.f32, ht->data.f32, rows * columns, 1e-5, "MPS logsumexp backward should match CPU");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hh);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(ht);
}

TEST_CASE("reduce logsumexp over multiple axes in NCHW on MPS")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_LOGSUMEXP_FORWARD, CCV_NNC_BACKEND_MPS));
	const float scale = 0.75f;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 3, 17, 19), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 3, 1, 1), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	int i;
	for (i = 0; i < 3 * 17 * 19; i++)
		ha->data.f32[i] = 40 * dsfmt_genrand_open_close(&dsfmt) - 20;
	ccv_nnc_cmd_exec(CMD_REDUCE_LOGSUMEXP_FORWARD(scale, 1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 3, 17, 19), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 3, 1, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_LOGSUMEXP_FORWARD(scale, 1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 3, 1, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hb->data.f32, bt->data.f32, 3, 1e-5, "multi-axis MPS logsumexp should match CPU");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("mps reduce logsumexp MFA matches MPSGraph over 200k-column partitioned rows")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_LOGSUMEXP_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 3;
	const int columns = 200003;
	const float scale = 0.625f;
	const int datatypes[] = { CCV_32F, CCV_16F, CCV_16BF };
	const uint64_t old_flags = ccv_nnc_flags();
	int d;
	for (d = 0; d < 3; d++)
	{
		const int datatype = datatypes[d];
		ccv_nnc_tensor_param_t host_input_params = CPU_TENSOR_NHWC(32F, rows, columns);
		host_input_params.datatype = datatype;
		ccv_nnc_tensor_param_t host_output_params = CPU_TENSOR_NHWC(32F, rows, 1);
		host_output_params.datatype = datatype;
		ccv_nnc_tensor_param_t input_params = host_input_params;
		input_params.type = CCV_TENSOR_GPU_MEMORY | 000;
		ccv_nnc_tensor_param_t output_params = host_output_params;
		output_params.type = CCV_TENSOR_GPU_MEMORY | 000;
		ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, host_input_params, 0);
		float* const values = (float*)ccmalloc(sizeof(float) * rows * columns);
		int i, j;
		for (i = 0; i < rows; i++)
		{
			for (j = 0; j < columns; j++)
				values[i * columns + j] = (float)((j * 7919 + i * 104729) % 8191) / 256 - 16;
			values[i * columns + (i * 65537 + 17) % columns] = 80 + i;
		}
		if (datatype == CCV_32F)
			memcpy(ha->data.f32, values, sizeof(float) * rows * columns);
		else if (datatype == CCV_16F)
			ccv_float_to_half_precision(values, (uint16_t*)ha->data.f16, rows * columns);
		else
			ccv_float_to_bfloat(values, (uint16_t*)ha->data.f16, rows * columns);
		ccfree(values);
		ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, input_params, 0);
		ccv_nnc_tensor_t* const graph_b = ccv_nnc_tensor_new(0, output_params, 0);
		ccv_nnc_tensor_t* const mfa_b = ccv_nnc_tensor_new(0, output_params, 0);
		ccv_nnc_tensor_t* const hgraph_b = ccv_nnc_tensor_new(0, host_output_params, 0);
		ccv_nnc_tensor_t* const hmfa_b = ccv_nnc_tensor_new(0, host_output_params, 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
		ccv_nnc_cmd_t cmd = CMD_REDUCE_LOGSUMEXP_FORWARD(scale, 1);
		cmd.backend = CCV_NNC_BACKEND_MPS;
		assert(cmd.backend >= 0);
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
		ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(graph_b), 0);
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
		ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(mfa_b), 0);
		if (old_flags & CCV_NNC_DISABLE_MFA)
			ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
		else
			ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(graph_b, mfa_b), TENSOR_LIST(hgraph_b, hmfa_b), 0);
		float graph_values[3];
		float mfa_values[3];
		if (datatype == CCV_32F)
		{
			memcpy(graph_values, hgraph_b->data.f32, sizeof(graph_values));
			memcpy(mfa_values, hmfa_b->data.f32, sizeof(mfa_values));
		} else if (datatype == CCV_16F) {
			ccv_half_precision_to_float((uint16_t*)hgraph_b->data.f16, graph_values, rows);
			ccv_half_precision_to_float((uint16_t*)hmfa_b->data.f16, mfa_values, rows);
		} else {
			ccv_bfloat_to_float((uint16_t*)hgraph_b->data.f16, graph_values, rows);
			ccv_bfloat_to_float((uint16_t*)hmfa_b->data.f16, mfa_values, rows);
		}
		const float tolerance = datatype == CCV_32F ? 2e-5 : (datatype == CCV_16F ? 0.04 : 0.3);
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, graph_values, mfa_values, rows, tolerance, "partitioned MFA logsumexp should match MPSGraph for %d", datatype);
		ccv_nnc_tensor_free(ha);
		ccv_nnc_tensor_free(a);
		ccv_nnc_tensor_free(graph_b);
		ccv_nnc_tensor_free(mfa_b);
		ccv_nnc_tensor_free(hgraph_b);
		ccv_nnc_tensor_free(hmfa_b);
	}
}

TEST_CASE("mps reduce logsumexp MFA matches MPSGraph special values across partitions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_LOGSUMEXP_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 5;
	const int columns = 8193;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, columns), 0);
	int i;
	for (i = 0; i < rows * columns; i++)
		ha->data.f32[i] = -INFINITY;
	for (i = 0; i < columns; i++)
		ha->data.f32[i] = NAN;
	for (i = 0; i < columns - 1; i++)
		ha->data.f32[2 * columns + i] = -4;
	ha->data.f32[3 * columns] = -3;
	ha->data.f32[4 * columns + 100] = 7;
	ha->data.f32[2 * columns + columns - 1] = NAN;
	ha->data.f32[3 * columns + columns - 1] = INFINITY;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, columns), 0);
	ccv_nnc_tensor_t* const graph_b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, 1), 0);
	ccv_nnc_tensor_t* const mfa_b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, 1), 0);
	ccv_nnc_tensor_t* const hgraph_b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, 1), 0);
	ccv_nnc_tensor_t* const hmfa_b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_t cmd = CMD_REDUCE_LOGSUMEXP_FORWARD(1, 1);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(graph_b), 0);
	ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(mfa_b), 0);
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(graph_b, mfa_b), TENSOR_LIST(hgraph_b, hmfa_b), 0);
	REQUIRE(isinf(hgraph_b->data.f32[0]) && hgraph_b->data.f32[0] < 0 && isinf(hmfa_b->data.f32[0]) && hmfa_b->data.f32[0] < 0, "an all-NaN row should reduce to negative infinity");
	REQUIRE(isinf(hgraph_b->data.f32[1]) && hgraph_b->data.f32[1] < 0 && isinf(hmfa_b->data.f32[1]) && hmfa_b->data.f32[1] < 0, "an all-negative-infinity row should remain negative infinity");
	REQUIRE(isnan(hgraph_b->data.f32[2]) && isnan(hmfa_b->data.f32[2]), "an all-NaN partition should poison a finite partition");
	REQUIRE(isinf(hgraph_b->data.f32[3]) && hgraph_b->data.f32[3] > 0 && isinf(hmfa_b->data.f32[3]) && hmfa_b->data.f32[3] > 0, "positive infinity should remain positive infinity");
	REQUIRE_EQ_WITH_TOLERANCE(hgraph_b->data.f32[4], hmfa_b->data.f32[4], 1e-6, "finite partitioned logsumexp should match MPSGraph");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(graph_b);
	ccv_nnc_tensor_free(mfa_b);
	ccv_nnc_tensor_free(hgraph_b);
	ccv_nnc_tensor_free(hmfa_b);
}

TEST_CASE("reduce max forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_MAX_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_MAX_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_MAX_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("mps reduce max MFA matches MPSGraph over one and multiple partitions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_MAX_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 7;
	const int datatypes[] = { CCV_32F, CCV_16F };
	const int columns_list[] = { 1023, 8193 };
	const uint64_t old_flags = ccv_nnc_flags();
	int d, s;
	for (d = 0; d < 2; d++)
		for (s = 0; s < 2; s++)
		{
			const int datatype = datatypes[d];
			const int columns = columns_list[s];
			ccv_nnc_tensor_param_t host_input_params = CPU_TENSOR_NHWC(32F, rows, columns);
			host_input_params.datatype = datatype;
			ccv_nnc_tensor_param_t host_output_params = CPU_TENSOR_NHWC(32F, rows, 1);
			host_output_params.datatype = datatype;
			ccv_nnc_tensor_param_t input_params = host_input_params;
			input_params.type = CCV_TENSOR_GPU_MEMORY | 000;
			ccv_nnc_tensor_param_t output_params = host_output_params;
			output_params.type = CCV_TENSOR_GPU_MEMORY | 000;
			ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, host_input_params, 0);
			float* const values = (float*)ccmalloc(sizeof(float) * rows * columns);
			float expected[7];
			int i, j;
			for (i = 0; i < rows; i++)
			{
				for (j = 0; j < columns; j++)
					values[i * columns + j] = (float)((j * 7919 + i * 104729) % 8191) / 1024 - 4;
				expected[i] = 32 + i;
				values[i * columns + (i * 1543 + 17) % columns] = expected[i];
			}
			if (datatype == CCV_32F)
				memcpy(ha->data.f32, values, sizeof(float) * rows * columns);
			else
				ccv_float_to_half_precision(values, (uint16_t*)ha->data.f16, rows * columns);
			ccfree(values);
			ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, input_params, 0);
			ccv_nnc_tensor_t* const graph_b = ccv_nnc_tensor_new(0, output_params, 0);
			ccv_nnc_tensor_t* const mfa_b = ccv_nnc_tensor_new(0, output_params, 0);
			ccv_nnc_tensor_t* const hgraph_b = ccv_nnc_tensor_new(0, host_output_params, 0);
			ccv_nnc_tensor_t* const hmfa_b = ccv_nnc_tensor_new(0, host_output_params, 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
			ccv_nnc_cmd_t cmd = CMD_REDUCE_MAX_FORWARD(1);
			cmd.backend = CCV_NNC_BACKEND_MPS;
			assert(cmd.backend >= 0);
			ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
			ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(graph_b), 0);
			ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
			ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(mfa_b), 0);
			if (old_flags & CCV_NNC_DISABLE_MFA)
				ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
			else
				ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(graph_b, mfa_b), TENSOR_LIST(hgraph_b, hmfa_b), 0);
			if (datatype == CCV_32F)
			{
				REQUIRE_ARRAY_EQ(float, hgraph_b->data.f32, hmfa_b->data.f32, rows, "MFA reduce max should match MPSGraph exactly for %d columns", columns);
				REQUIRE_ARRAY_EQ(float, expected, hmfa_b->data.f32, rows, "MFA reduce max should select the known maxima for %d columns", columns);
			} else {
				uint16_t expected_half[7];
				ccv_float_to_half_precision(expected, expected_half, rows);
				REQUIRE_ARRAY_EQ(uint16_t, (uint16_t*)hgraph_b->data.f16, (uint16_t*)hmfa_b->data.f16, rows, "half-precision MFA reduce max should match MPSGraph exactly for %d columns", columns);
				REQUIRE_ARRAY_EQ(uint16_t, expected_half, (uint16_t*)hmfa_b->data.f16, rows, "half-precision MFA reduce max should select the known maxima for %d columns", columns);
			}
			ccv_nnc_tensor_free(ha);
			ccv_nnc_tensor_free(a);
			ccv_nnc_tensor_free(graph_b);
			ccv_nnc_tensor_free(mfa_b);
			ccv_nnc_tensor_free(hgraph_b);
			ccv_nnc_tensor_free(hmfa_b);
		}
}

TEST_CASE("mps reduce max MFA matches MPSGraph NaN behavior")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_MAX_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 4;
	const int columns = 257;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, columns), 0);
	int i;
	for (i = 0; i < rows * columns; i++)
		ha->data.f32[i] = -INFINITY;
	for (i = 0; i < columns; i++)
		ha->data.f32[i] = NAN;
	ha->data.f32[columns] = NAN;
	ha->data.f32[columns + 19] = 7;
	ha->data.f32[2 * columns + 113] = NAN;
	ha->data.f32[2 * columns + 256] = 11;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, columns), 0);
	ccv_nnc_tensor_t* const graph_b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, 1), 0);
	ccv_nnc_tensor_t* const mfa_b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, 1), 0);
	ccv_nnc_tensor_t* const hgraph_b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, 1), 0);
	ccv_nnc_tensor_t* const hmfa_b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_t cmd = CMD_REDUCE_MAX_FORWARD(1);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(graph_b), 0);
	ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(mfa_b), 0);
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(graph_b, mfa_b), TENSOR_LIST(hgraph_b, hmfa_b), 0);
	REQUIRE_ARRAY_EQ(uint32_t, (uint32_t*)hgraph_b->data.f32, (uint32_t*)hmfa_b->data.f32, rows, "MFA reduce max should match MPSGraph NaN behavior exactly");
	REQUIRE(isinf(hmfa_b->data.f32[0]) && hmfa_b->data.f32[0] < 0, "an all-NaN row should reduce to negative infinity");
	REQUIRE_EQ(7, hmfa_b->data.f32[1], "NaNs should be ignored when a finite maximum exists");
	REQUIRE_EQ(11, hmfa_b->data.f32[2], "a later NaN should be ignored when a finite maximum exists");
	REQUIRE(isinf(hmfa_b->data.f32[3]) && hmfa_b->data.f32[3] < 0, "an all-negative-infinity row should remain negative infinity");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(graph_b);
	ccv_nnc_tensor_free(mfa_b);
	ccv_nnc_tensor_free(hgraph_b);
	ccv_nnc_tensor_free(hmfa_b);
}

TEST_CASE("reduce min forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_MIN_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_MIN_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_MIN_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("argmin with float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ARGMIN_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_ARGMIN_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 3, 5, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10, 1, 5, 3), 0);
	int i;
	for (i = 0; i < 10 * 3 * 5 * 3; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_ARGMIN_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 3, 5, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 10, 1, 5, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_ARGMIN_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10, 1, 5, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("argmax with float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ARGMAX_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_ARGMAX_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 3, 5, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10, 1, 5, 3), 0);
	int i;
	for (i = 0; i < 10 * 3 * 5 * 3; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_ARGMAX_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 3, 5, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 10, 1, 5, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_ARGMAX_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10, 1, 5, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("argmax with bfloat")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ARGMAX_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_ARGMAX_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const ha32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 3, 5, 3), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 10, 3, 5, 3), 0);
	ccv_nnc_tensor_t* const hrounded = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 3, 5, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10, 1, 5, 3), 0);
	int i;
	for (i = 0; i < 10 * 3 * 5 * 3; i++)
		ha32->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_float_to_bfloat(ha32->data.f32, (uint16_t*)ha->data.f16, 10 * 3 * 5 * 3);
	ccv_bfloat_to_float((uint16_t*)ha->data.f16, hrounded->data.f32, 10 * 3 * 5 * 3);
	ccv_nnc_cmd_exec(CMD_ARGMAX_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(hrounded), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 10, 3, 5, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 10, 1, 5, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_ARGMAX_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10, 1, 5, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha32);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hrounded);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("argmax with bfloat over large axis")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ARGMAX_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_ARGMAX_FORWARD, CCV_NNC_BACKEND_MPS));
	const int length = 2048;
	const int winner = 1500;
	ccv_nnc_tensor_t* const ha32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, length), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 1, length), 0);
	ccv_nnc_tensor_t* const hrounded = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, length), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1, 1), 0);
	int i;
	for (i = 0; i < length; i++)
		ha32->data.f32[i] = -2;
	ha32->data.f32[0] = 9.875;
	ha32->data.f32[599] = 19.25;
	ha32->data.f32[1044] = 19.5;
	ha32->data.f32[winner] = 19.625;
	ccv_float_to_bfloat(ha32->data.f32, (uint16_t*)ha->data.f16, length);
	ccv_bfloat_to_float((uint16_t*)ha->data.f16, hrounded->data.f32, length);
	ccv_nnc_cmd_exec(CMD_ARGMAX_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(hrounded), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 1, length), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 1, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_ARGMAX_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	REQUIRE_EQ(winner, bt->data.i32[0], "large-axis argmax should find winner");
	ccv_nnc_tensor_free(ha32);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hrounded);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("mps argmax over multiple MFA partitions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ARGMAX_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 5;
	const int columns = 8193;
	const int winners[] = { 0, 4095, 4096, 8191, 8192 };
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, columns), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, rows, 1), 0);
	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < columns; j++)
			ha->data.f32[i * columns + j] = -2;
		ha->data.f32[i * columns + winners[i]] = 10;
	}
	ccv_nnc_cmd_exec(CMD_ARGMAX_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, columns), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, rows, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_t cmd = CMD_ARGMAX_FORWARD(1);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, rows, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "partitioned MFA argmax should match CPU");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("mps gumbel argmax one-pass MFA is reproducible")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GUMBEL_ARGMAX_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 64;
	const int columns = 257;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, columns), 0);
	int i;
	for (i = 0; i < rows * columns; i++)
		ha->data.f32[i] = 0;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, columns), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, rows, 1), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, rows, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_tensor_t* const ha_strided = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, columns + 1), 0);
	ccv_nnc_tensor_t* const a_strided = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, columns + 1), 0);
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a_strided, GPU_TENSOR_NHWC(000, 32F, rows, columns), ccv_nnc_no_ofs, DIM_ALLOC(columns + 1, 1));
	ccv_nnc_tensor_t* const d = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, rows, 1), 0);
	ccv_nnc_tensor_t* const e = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, rows, 1), 0);
	for (i = 0; i < rows * (columns + 1); i++)
		ha_strided->data.f32[i] = 0;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha_strided), TENSOR_LIST(a_strided), 0);
	ccv_nnc_stream_context_t* const stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_cmd_t cmd = CMD_GUMBEL_ARGMAX(1, 1);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	ccv_nnc_stream_context_set_seed(stream_context, 177);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), stream_context);
	ccv_nnc_stream_context_set_seed(stream_context, 177);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(c), stream_context);
	ccv_nnc_stream_context_set_seed(stream_context, 177);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av), TENSOR_LIST(d), stream_context);
	ccv_nnc_stream_context_set_seed(stream_context, 177);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av), TENSOR_LIST(e), stream_context);
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, rows, 1), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, rows, 1), 0);
	ccv_nnc_tensor_t* const hd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, rows, 1), 0);
	ccv_nnc_tensor_t* const he = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, rows, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, c, d, e), TENSOR_LIST(hb, hc, hd, he), 0);
	REQUIRE_TENSOR_EQ(hb, hc, "the same stream seed should reproduce MFA gumbel argmax");
	REQUIRE_TENSOR_EQ(hd, he, "the same stream seed should reproduce MPSGraph gumbel argmax");
	int different = 0;
	for (i = 0; i < rows; i++)
	{
		REQUIRE(hb->data.i32[i] >= 0 && hb->data.i32[i] < columns, "MFA gumbel argmax should return a valid index");
		different |= hb->data.i32[i] != hb->data.i32[0];
	}
	REQUIRE(different, "equal logits should sample more than one category");
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha_strided);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(a_strided);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(e);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hd);
	ccv_nnc_tensor_free(he);
}

TEST_CASE("mps gumbel argmax specializes the MFA scale function constant")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GUMBEL_ARGMAX_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 32;
	const int columns = 257;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, columns), 0);
	int i, j;
	for (i = 0; i < rows; i++)
		for (j = 0; j < columns; j++)
			ha->data.f32[i * columns + j] = j == i ? 1 : 0;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, columns), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, rows, 1), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, rows, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_stream_context_t* const stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_cmd_t cmd = CMD_GUMBEL_ARGMAX(1000, 1);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	ccv_nnc_stream_context_set_seed(stream_context, 227);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), stream_context);
	cmd = CMD_GUMBEL_ARGMAX(0, 1);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	ccv_nnc_stream_context_set_seed(stream_context, 227);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), stream_context);
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	for (i = 0; i < rows; i++)
		REQUIRE_EQ(i, hb->data.i32[i], "zero Gumbel scale should reduce to deterministic argmax after another scale was cached");
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(hb);
}

TEST_CASE("mps gumbel argmax over multiple MFA partitions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GUMBEL_ARGMAX_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 4;
	const int columns = 8193;
	const int winners[] = { 17, 4095, 4096, 8192 };
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, columns), 0);
	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < columns; j++)
			ha->data.f32[i * columns + j] = -100;
		ha->data.f32[i * columns + winners[i]] = 100;
	}
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, columns), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, rows, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_stream_context_t* const stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_stream_context_set_seed(stream_context, 31);
	ccv_nnc_cmd_t cmd = CMD_GUMBEL_ARGMAX(1, 1);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), stream_context);
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, rows, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	for (i = 0; i < rows; i++)
		REQUIRE_EQ(winners[i], hb->data.i32[i], "partitioned MFA gumbel argmax should find the dominant logit");
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(hb);
}

TEST_CASE("mps gumbel argmax falls back for a non-final axis")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GUMBEL_ARGMAX_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batches = 2;
	const int categories = 3;
	const int width = 5;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batches, categories, width), 0);
	int i, j, k;
	for (i = 0; i < batches; i++)
		for (j = 0; j < categories; j++)
			for (k = 0; k < width; k++)
				ha->data.f32[(i * categories + j) * width + k] = j == 1 ? 100 : -100;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batches, categories, width), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, batches, 1, width), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_stream_context_t* const stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_stream_context_set_seed(stream_context, 47);
	ccv_nnc_cmd_t cmd = CMD_GUMBEL_ARGMAX(1, 1);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), stream_context);
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, batches, 1, width), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	for (i = 0; i < batches * width; i++)
		REQUIRE_EQ(1, hb->data.i32[i], "MPSGraph gumbel argmax should find the dominant logit");
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(hb);
}

TEST_CASE("reduce norm2 forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_NORM2_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_NORM2_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("reduce norm2 forward noop")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_NORM2_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_NORM2_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("reduce norm2 backward")
{
	GUARD_ELSE_RETURN((ccv_nnc_cmd_ok(CCV_NNC_REDUCE_NORM2_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_NORM2_FORWARD, CCV_NNC_BACKEND_MPS)) &&
		(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_NORM2_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_NORM2_FORWARD, CCV_NNC_BACKEND_MPS)));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const hh = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	hg->data.f32[0] = 1;
	hg->data.f32[1] = 2;
	hg->data.f32[2] = 3;
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_BACKWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hb), TENSOR_LIST(hh), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg), TENSOR_LIST(g), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_BACKWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(h), 0);
	ccv_nnc_tensor_t* const ht = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(h), TENSOR_LIST(ht), 0);
	REQUIRE_TENSOR_EQ(hh, ht, "result should be equal");
	ccv_nnc_tensor_free(hh);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ht);
}

TEST_CASE("reduce isnan float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_ISNAN_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_ISNAN_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	ha->data.f32[0] = NAN;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_ISNAN_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_ISNAN_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("reduce isnan in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_ISNAN_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_ISNAN_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
	ha->data.f32[0] = NAN;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_tensor_t* const ha16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(ha16), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_ISNAN_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16), TENSOR_LIST(a16), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_ISNAN_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a16), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha16);
	ccv_nnc_tensor_free(a16);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

#include "case_main.h"
