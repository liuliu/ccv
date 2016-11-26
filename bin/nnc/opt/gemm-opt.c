#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>
#include <sys/time.h>
#include <ctype.h>

static unsigned int get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

#define INPUT_DIM (4096)
#define OUTPUT_DIM (4096)

int main(int argc, char** argv)
{
	ccv_nnc_init();
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(INPUT_DIM), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(OUTPUT_DIM), 0);
	ccv_nnc_cmd_t forw_cmd = ccv_nnc_cmd(CCV_NNC_GEMM_FORWARD, 0, CMD_GEMM(OUTPUT_DIM), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(INPUT_DIM, OUTPUT_DIM), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(OUTPUT_DIM), 0);
	// configure the inlets.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < INPUT_DIM * OUTPUT_DIM; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / INPUT_DIM;
	for (i = 0; i < INPUT_DIM; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < OUTPUT_DIM; i++)
		bias->data.f32[i] = (float)i / OUTPUT_DIM;
	unsigned int elapsed_time = get_current_time();
	ccv_nnc_cmd_exec(forw_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	elapsed_time = get_current_time() - elapsed_time;
	printf("forw %u ms for ref\n", elapsed_time);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(OUTPUT_DIM), 0);
	forw_cmd.backend = CCV_NNC_BACKEND_CPU_OPT;
	assert(forw_cmd.backend >= 0);
	forw_cmd.algorithm = 0; // CCV_NNC_CMD_OPT_FC_ALGO_DIRECT = 0
	elapsed_time = get_current_time();
	ccv_nnc_cmd_exec(forw_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(c), 0);
	elapsed_time = get_current_time() - elapsed_time;
	printf("forw %u ms for optimized\n", elapsed_time);
	for (i = 0; i < OUTPUT_DIM; i++)
		if (fabs(b->data.f32[i] - c->data.f32[i]) > 1e-5)
			printf("forw output[%d]: %f %f\n", i, b->data.f32[i], c->data.f32[i]);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(INPUT_DIM, OUTPUT_DIM), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(INPUT_DIM), 0);
	ccv_nnc_cmd_t back_cmd = ccv_nnc_cmd(CCV_NNC_GEMM_BACKWARD, 0, CMD_GEMM(OUTPUT_DIM), 0);
	elapsed_time = get_current_time();
	ccv_nnc_cmd_exec(back_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(b, a, w), TENSOR_LIST(h, dw, bias), 0);
	elapsed_time = get_current_time() - elapsed_time;
	printf("back %u ms for ref\n", elapsed_time);
	ccv_nnc_tensor_t* dwc = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(INPUT_DIM, OUTPUT_DIM), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(INPUT_DIM), 0);
	ccv_nnc_tensor_t* biasc = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(OUTPUT_DIM), 0);
	back_cmd.backend = CCV_NNC_BACKEND_CPU_OPT;
	assert(back_cmd.backend >= 0);
	back_cmd.algorithm = 0; // CCV_NNC_CMD_OPT_FC_ALGO_DIRECT = 0
	elapsed_time = get_current_time();
	ccv_nnc_cmd_exec(back_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(b, a, w), TENSOR_LIST(hc, dwc, biasc), 0);
	elapsed_time = get_current_time() - elapsed_time;
	printf("back %u ms for optimized\n", elapsed_time);
	for (i = 0; i < OUTPUT_DIM; i++)
		if (fabs(bias->data.f32[i] - biasc->data.f32[i]) > 1e-5)
			printf("back bias[%d]: %f %f\n", i, bias->data.f32[i], biasc->data.f32[i]);
	for (i = 0; i < INPUT_DIM * OUTPUT_DIM; i++)
		if (fabs(dw->data.f32[i] - dwc->data.f32[i]) > 1e-5)
			printf("back dw[%d]: %f %f\n", i, dw->data.f32[i], dwc->data.f32[i]);
	for (i = 0; i < INPUT_DIM; i++)
		if (fabs(h->data.f32[i] - hc->data.f32[i]) > 1e-5)
			printf("back h[%d]: %f %f\n", i, h->data.f32[i], hc->data.f32[i]);
	ccv_nnc_tensor_free(biasc);
	ccv_nnc_tensor_free(dwc);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}
