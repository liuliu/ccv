#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <sys/time.h>
#include <ctype.h>

static unsigned int get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

#define DIM (512)

#define SIZE (58)

int main(int argc, char** argv)
{
	ccv_nnc_init();
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(DIM, SIZE, SIZE), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(DIM, SIZE, SIZE), 0);
	ccv_nnc_cmd_t cmd = ccv_nnc_cmd(CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD, 0, CMD_CONVOLUTIONAL(DIM, DIM, 3, 3), 0);
	ccv_nnc_hint_t hint = ccv_nnc_hint_guess(cmd.info, &a->info, 1, &b->info, 1);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(DIM, 3, 3, DIM), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(DIM), 0);
	// configure the inlets.
	int i;
	for (i = 0; i < DIM * 3 * 3 * DIM; i++)
		w->data.f32[i] = (float)i / 512;
	for (i = 0; i < SIZE * SIZE * DIM; i++)
		a->data.f32[i] = (float)i / 1024;
	for (i = 0; i < DIM; i++)
		bias->data.f32[i] = i;
	unsigned int elapsed_time = get_current_time();
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b));
	elapsed_time = get_current_time() - elapsed_time;
	printf("%u ms for ref impl\n", elapsed_time);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(DIM, SIZE, SIZE), 0);
	cmd.backend = 0; // CCV_NNC_BACKEND_CPU_OPT = 0
	elapsed_time = get_current_time();
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(c));
	elapsed_time = get_current_time() - elapsed_time;
	printf("%u ms for winograd\n", elapsed_time);
	//for (i = 0; i < DIM * SIZE * SIZE; i++)
	//	printf("%f %f\n", b->data.f32[i], c->data.f32[i]);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}
