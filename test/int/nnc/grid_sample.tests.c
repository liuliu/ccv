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

TEST_CASE("compare grid_sample in NCHW with mps / cudnn with align corners")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GRID_SAMPLE_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) ||
		ccv_nnc_cmd_ok(CCV_NNC_GRID_SAMPLE_FORWARD, CCV_NNC_BACKEND_MPS));
	const int N = 2;
	const int C = 13;
	const int H = 15;
	const int W = 17;
	const int H_OUT = 7;
	const int W_OUT = 9;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, N, C, H, W), 0);
	ccv_nnc_tensor_t* const grid = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, N, H_OUT, W_OUT, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, N, C, H_OUT, W_OUT), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, N, C, H, W), 0);
	ccv_nnc_tensor_t* const hgrid = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, N, H_OUT, W_OUT, 2), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, N, C, H_OUT, W_OUT), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, N, C, H_OUT, W_OUT), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	const int input_count = N * C * H * W;
	const int grid_count = N * H_OUT * W_OUT * 2;
	const int output_count = N * C * H_OUT * W_OUT;
	int i;
	for (i = 0; i < input_count; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	for (i = 0; i < grid_count; i++)
		hgrid->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2.4 - 1.2;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hgrid), TENSOR_LIST(grid), 0);
	ccv_nnc_cmd_exec(CMD_GRID_SAMPLE_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a, grid), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GRID_SAMPLE_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hgrid), TENSOR_LIST(hbt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hbt->data.f32, hb->data.f32, output_count, 1e-4, "GPU computed output should match CPU reference implementation");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(grid);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hgrid);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("compare grid_sample in NCHW with mps / cudnn")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GRID_SAMPLE_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) ||
		ccv_nnc_cmd_ok(CCV_NNC_GRID_SAMPLE_FORWARD, CCV_NNC_BACKEND_MPS));
	const int N = 2;
	const int C = 13;
	const int H = 15;
	const int W = 17;
	const int H_OUT = 7;
	const int W_OUT = 9;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, N, C, H, W), 0);
	ccv_nnc_tensor_t* const grid = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, N, H_OUT, W_OUT, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, N, C, H_OUT, W_OUT), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, N, C, H, W), 0);
	ccv_nnc_tensor_t* const hgrid = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, N, H_OUT, W_OUT, 2), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, N, C, H_OUT, W_OUT), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, N, C, H_OUT, W_OUT), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 3);
	const int input_count = N * C * H * W;
	const int grid_count = N * H_OUT * W_OUT * 2;
	const int output_count = N * C * H_OUT * W_OUT;
	int i;
	for (i = 0; i < input_count; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	for (i = 0; i < grid_count; i++)
		hgrid->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2.4 - 1.2;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hgrid), TENSOR_LIST(grid), 0);
	ccv_nnc_cmd_exec(CMD_GRID_SAMPLE_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a, grid), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GRID_SAMPLE_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hgrid), TENSOR_LIST(hbt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hbt->data.f32, hb->data.f32, output_count, 1e-4, "GPU computed output should match CPU reference implementation");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(grid);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hgrid);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("compare grid_sample in NHWC with mps / cudnn with align corners")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GRID_SAMPLE_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) ||
		ccv_nnc_cmd_ok(CCV_NNC_GRID_SAMPLE_FORWARD, CCV_NNC_BACKEND_MPS));
	const int N = 2;
	const int C = 11;
	const int H = 14;
	const int W = 18;
	const int H_OUT = 6;
	const int W_OUT = 10;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, N, H, W, C), 0);
	ccv_nnc_tensor_t* const grid = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, N, H_OUT, W_OUT, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, N, H_OUT, W_OUT, C), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, N, H, W, C), 0);
	ccv_nnc_tensor_t* const hgrid = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, N, H_OUT, W_OUT, 2), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, N, H_OUT, W_OUT, C), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, N, H_OUT, W_OUT, C), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 2);
	const int input_count = N * C * H * W;
	const int grid_count = N * H_OUT * W_OUT * 2;
	const int output_count = N * C * H_OUT * W_OUT;
	int i;
	for (i = 0; i < input_count; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	for (i = 0; i < grid_count; i++)
		hgrid->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2.4 - 1.2;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hgrid), TENSOR_LIST(grid), 0);
	ccv_nnc_cmd_exec(CMD_GRID_SAMPLE_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a, grid), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GRID_SAMPLE_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hgrid), TENSOR_LIST(hbt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hbt->data.f32, hb->data.f32, output_count, 1e-4, "GPU computed output should match CPU reference implementation");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(grid);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hgrid);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("compare grid_sample in NHWC with mps / cudnn")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GRID_SAMPLE_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) ||
		ccv_nnc_cmd_ok(CCV_NNC_GRID_SAMPLE_FORWARD, CCV_NNC_BACKEND_MPS));
	const int N = 2;
	const int C = 11;
	const int H = 14;
	const int W = 18;
	const int H_OUT = 6;
	const int W_OUT = 10;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, N, H, W, C), 0);
	ccv_nnc_tensor_t* const grid = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, N, H_OUT, W_OUT, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, N, H_OUT, W_OUT, C), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, N, H, W, C), 0);
	ccv_nnc_tensor_t* const hgrid = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, N, H_OUT, W_OUT, 2), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, N, H_OUT, W_OUT, C), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, N, H_OUT, W_OUT, C), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 4);
	const int input_count = N * C * H * W;
	const int grid_count = N * H_OUT * W_OUT * 2;
	const int output_count = N * C * H_OUT * W_OUT;
	int i;
	for (i = 0; i < input_count; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	for (i = 0; i < grid_count; i++)
		hgrid->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2.4 - 1.2;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hgrid), TENSOR_LIST(grid), 0);
	ccv_nnc_cmd_exec(CMD_GRID_SAMPLE_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a, grid), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GRID_SAMPLE_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hgrid), TENSOR_LIST(hbt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hbt->data.f32, hb->data.f32, output_count, 1e-4, "GPU computed output should match CPU reference implementation");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(grid);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hgrid);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hbt);
}

#include "case_main.h"
