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

static int weight_dim(int bidirectional, int num_layers, int input_size, int hidden_size, int proj_size, int bias)
{
	const int D = !!bidirectional + 1;
	if (hidden_size == proj_size)
		return (num_layers * (bias ? 8 : 0) + (num_layers - 1) * (hidden_size * 4 * D + hidden_size * 4) + input_size * 4 + hidden_size * 4) * D;
	else
		return (num_layers * (bias ? 8 : 0) + (num_layers - 1) * (proj_size * 4 * D + proj_size * 4) + (proj_size * 4 + input_size * 4) + num_layers * proj_size) * D;
}

static int r_dim(int bidirectional, int dropout, int batch_count, int num_layers, int max_seq_count, int hidden_size, int proj_size)
{
	const int D = !!bidirectional + 1;
	if (hidden_size == proj_size)
		// 5: i, f, g, o, tanh(c)
		// 2: c, h
		return D * batch_count * ((5 + !!dropout) * num_layers * max_seq_count + 2 * num_layers * (max_seq_count - 1));
	else
		// 6: i, f, g, o, tanh(c), h
		// 1: c, h_proj
		return D * batch_count * ((6 + !!dropout) * num_layers * max_seq_count + 2 * num_layers * (max_seq_count - 1));
}

TEST_CASE("LSTM forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_LSTM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 1, 24), 0);
	ccv_nnc_tensor_t* const hx = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1, 24), 0);
	ccv_nnc_tensor_t* const cx = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1, 24), 0);
	const int weight_d = weight_dim(0, 6, 24, 24, 24, 1);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, weight_d, 24), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 1, 24), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1, 24), 0);
	ccv_nnc_tensor_t* const cy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1, 24), 0);
	const int r_d = r_dim(0, 0, 1, 6, 5, 24, 24);
	ccv_nnc_tensor_t* const r = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, r_d, 24), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 5 * 24; i++)
		x->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 3 * 12; i++)
		hx->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 3 * 24; i++)
		cx->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 24 * weight_d; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const gx = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5, 1, 24), 0);
	ccv_nnc_tensor_t* const ghx = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1, 24), 0);
	ccv_nnc_tensor_t* const gcx = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1, 24), 0);
	ccv_nnc_tensor_t* const gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, weight_d, 24), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x, hx, cx, w), TENSOR_LIST(gx, ghx, gcx, gw), 0);
	ccv_nnc_tensor_t* const gy = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5, 1, 24), 0);
	ccv_nnc_tensor_t* const ghy = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1, 24), 0);
	ccv_nnc_tensor_t* const gcy = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1, 24), 0);
	ccv_nnc_tensor_t* const gr = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, r_d, 24), 0);
	ccv_nnc_cmd_exec(CMD_LSTM_FORWARD(24, 0, 6, 1, 0, 0, 0, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gx, 0, ghx, gcx, gw), TENSOR_LIST(gy, ghy, gcy, gr), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gy, ghy, gcy, gr), TENSOR_LIST(y, hy, cy, r), 0);
	ccv_nnc_tensor_free(gx);
	ccv_nnc_tensor_free(ghx);
	ccv_nnc_tensor_free(gcx);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(gy);
	ccv_nnc_tensor_free(ghy);
	ccv_nnc_tensor_free(gcy);
	ccv_nnc_tensor_free(gr);
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(hx);
	ccv_nnc_tensor_free(cx);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(cy);
	ccv_nnc_tensor_free(r);
}

TEST_CASE("LSTM forward with dropout")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_LSTM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 1, 24), 0);
	ccv_nnc_tensor_t* const hx = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1, 24), 0);
	ccv_nnc_tensor_t* const cx = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1, 24), 0);
	const int weight_d = weight_dim(0, 6, 24, 24, 24, 1);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, weight_d, 24), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 1, 24), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1, 24), 0);
	ccv_nnc_tensor_t* const cy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1, 24), 0);
	const int r_d = r_dim(0, 1, 1, 6, 5, 24, 24);
	ccv_nnc_tensor_t* const r = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, r_d, 24), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 5 * 24; i++)
		x->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 3 * 12; i++)
		hx->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 3 * 24; i++)
		cx->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 24 * weight_d; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const gx = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5, 1, 24), 0);
	ccv_nnc_tensor_t* const ghx = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1, 24), 0);
	ccv_nnc_tensor_t* const gcx = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1, 24), 0);
	ccv_nnc_tensor_t* const gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, weight_d, 24), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x, hx, cx, w), TENSOR_LIST(gx, ghx, gcx, gw), 0);
	ccv_nnc_tensor_t* const gy = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5, 1, 24), 0);
	ccv_nnc_tensor_t* const ghy = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1, 24), 0);
	ccv_nnc_tensor_t* const gcy = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1, 24), 0);
	ccv_nnc_tensor_t* const gr = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, r_d, 24), 0);
	ccv_nnc_cmd_exec(CMD_LSTM_FORWARD(24, 0, 6, 1, 0, 0, 0.5, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gx, 0, ghx, gcx, gw), TENSOR_LIST(gy, ghy, gcy, gr), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gy, ghy, gcy, gr), TENSOR_LIST(y, hy, cy, r), 0);
	ccv_nnc_tensor_free(gx);
	ccv_nnc_tensor_free(ghx);
	ccv_nnc_tensor_free(gcx);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(gy);
	ccv_nnc_tensor_free(ghy);
	ccv_nnc_tensor_free(gcy);
	ccv_nnc_tensor_free(gr);
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(hx);
	ccv_nnc_tensor_free(cx);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(cy);
	ccv_nnc_tensor_free(r);
}

TEST_CASE("LSTM forward with projection")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_LSTM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 1, 24), 0);
	ccv_nnc_tensor_t* const hx = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1, 12), 0);
	ccv_nnc_tensor_t* const cx = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1, 24), 0);
	const int weight_d = weight_dim(0, 6, 24, 24, 12, 1);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, weight_d, 24), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 1, 12), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1, 12), 0);
	ccv_nnc_tensor_t* const cy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1, 24), 0);
	const int r_d = r_dim(0, 1, 1, 6, 5, 24, 12);
	ccv_nnc_tensor_t* const r = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, r_d, 24), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 5 * 24; i++)
		x->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 3 * 12; i++)
		hx->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 3 * 24; i++)
		cx->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 24 * weight_d; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const gx = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5, 1, 24), 0);
	ccv_nnc_tensor_t* const ghx = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1, 12), 0);
	ccv_nnc_tensor_t* const gcx = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1, 24), 0);
	ccv_nnc_tensor_t* const gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, weight_d, 24), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x, hx, cx, w), TENSOR_LIST(gx, ghx, gcx, gw), 0);
	ccv_nnc_tensor_t* const gy = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5, 1, 12), 0);
	ccv_nnc_tensor_t* const ghy = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1, 12), 0);
	ccv_nnc_tensor_t* const gcy = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1, 24), 0);
	ccv_nnc_tensor_t* const gr = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, r_d, 24), 0);
	ccv_nnc_cmd_exec(CMD_LSTM_FORWARD(24, 12, 6, 1, 0, 0, 0.5, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gx, 0, ghx, gcx, gw), TENSOR_LIST(gy, ghy, gcy, gr), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gy, ghy, gcy, gr), TENSOR_LIST(y, hy, cy, r), 0);
	ccv_nnc_tensor_free(gx);
	ccv_nnc_tensor_free(ghx);
	ccv_nnc_tensor_free(gcx);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(gy);
	ccv_nnc_tensor_free(ghy);
	ccv_nnc_tensor_free(gcy);
	ccv_nnc_tensor_free(gr);
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(hx);
	ccv_nnc_tensor_free(cx);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(cy);
	ccv_nnc_tensor_free(r);
}

TEST_CASE("LSTM forward with projection, bidirectional")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_LSTM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 1, 24), 0);
	ccv_nnc_tensor_t* const hx = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 12, 1, 12), 0);
	ccv_nnc_tensor_t* const cx = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 12, 1, 24), 0);
	const int weight_d = weight_dim(1, 6, 24, 24, 12, 1);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, weight_d, 24), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 1, 24), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 12, 1, 12), 0);
	ccv_nnc_tensor_t* const cy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 12, 1, 24), 0);
	const int r_d = r_dim(1, 1, 1, 6, 5, 24, 12);
	ccv_nnc_tensor_t* const r = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, r_d, 24), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 5 * 24; i++)
		x->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 3 * 12; i++)
		hx->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 3 * 24; i++)
		cx->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 24 * weight_d; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const gx = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5, 1, 24), 0);
	ccv_nnc_tensor_t* const ghx = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 12, 1, 12), 0);
	ccv_nnc_tensor_t* const gcx = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 12, 1, 24), 0);
	ccv_nnc_tensor_t* const gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, weight_d, 24), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x, hx, cx, w), TENSOR_LIST(gx, ghx, gcx, gw), 0);
	ccv_nnc_tensor_t* const gy = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5, 1, 24), 0);
	ccv_nnc_tensor_t* const ghy = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 12, 1, 12), 0);
	ccv_nnc_tensor_t* const gcy = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 12, 1, 24), 0);
	ccv_nnc_tensor_t* const gr = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, r_d, 24), 0);
	ccv_nnc_cmd_exec(CMD_LSTM_FORWARD(24, 12, 6, 1, 0, 1, 0.5, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gx, 0, ghx, gcx, gw), TENSOR_LIST(gy, ghy, gcy, gr), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gy, ghy, gcy, gr), TENSOR_LIST(y, hy, cy, r), 0);
	ccv_nnc_tensor_free(gx);
	ccv_nnc_tensor_free(ghx);
	ccv_nnc_tensor_free(gcx);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(gy);
	ccv_nnc_tensor_free(ghy);
	ccv_nnc_tensor_free(gcy);
	ccv_nnc_tensor_free(gr);
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(hx);
	ccv_nnc_tensor_free(cx);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(cy);
	ccv_nnc_tensor_free(r);
}

TEST_CASE("LSTM backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_LSTM_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN));
}

#include "case_main.h"
