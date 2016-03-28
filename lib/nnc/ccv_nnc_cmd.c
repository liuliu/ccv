#include "ccv_nnc.h"

typedef void(*ccv_nnc_init_f)(ccv_nnc_cmd_api_t cmd_api[]);

typedef struct {
	int backend;
	ccv_nnc_init_f init;
} ccv_nnc_init_t;

#define CCV_NNC_INIT_DECL(init_func) extern void (init_func)(ccv_nnc_cmd_api_t cmd_api[])
#define CCV_NNC_INIT_MAP_BEGIN() static ccv_nnc_init_t init_map[] = {
#define CCV_NNC_INIT_MAP(name, init_func) { .backend = name, .init = init_func, },
#define CCV_NNC_INIT_MAP_END() };

#define CCV_NNC_INIT_EXEC(name, init_func) do { \
		(init_func)(command_api_decls[name]); \
	} while (0)

void ccv_nnc_gpu_ref_init(ccv_nnc_cmd_api_t cmd_api[])
{
}

void ccv_nnc_gpu_cudnn_init(ccv_nnc_cmd_api_t cmd_api[])
{
}

// I should be able to automatically extract code below from source code.
#include "ccv_nnc_init.inc"
// Above should be automatic generated.

static ccv_nnc_cmd_api_t cmd_api_decls[CCV_NNC_BACKEND_COUNT][CCV_NNC_COMPUTE_COUNT];

void ccv_nnc_init(void)
{
	int i;
	int count = sizeof(init_map) / sizeof(ccv_nnc_init_t);
	// Init dynamic dispatch table.
	for (i = 0; i < count; i++)
		init_map[i].init(cmd_api_decls[init_map[i].backend]);
}

const ccv_nnc_cmd_param_t ccv_nnc_default_cmd_params = {{{0}}};

ccv_nnc_cmd_t ccv_nnc_cmd(const int compute, ccv_nnc_cmd_exec_f exec, const ccv_nnc_cmd_param_t params, const int flags)
{
	ccv_nnc_cmd_t cmd;
	cmd.info = params;
	// TODO: auto-find a workable implementation.
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert((compute == CCV_NNC_COMPUTE_CUSTOM && exec) || (compute != CCV_NNC_COMPUTE_CUSTOM && !exec));
	cmd.compute = compute;
	cmd.exec = exec;
	return cmd;
}

const ccv_nnc_hint_t ccv_nnc_default_hint = {{{0}}};

int ccv_nnc_hint_verify(const ccv_nnc_hint_t hint, const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t a, const ccv_nnc_tensor_param_t b)
{
	int i;
	// 0-dim is reserved for channels
	for (i = 1; i < CCV_NNC_MAX_DIM + 1; i++)
	{
		if ((hint.border.begin[i] + hint.border.end[i] + a.dim[i] - cmd.size.dim[i]) % hint.stride.dim[i] != 0)
			return -1;
		int expected = (hint.border.begin[i] + hint.border.end[i] + a.dim[i] - cmd.size.dim[i]) / hint.stride.dim[i] + 1;
		if (expected != b.dim[i])
			return -1;
	}
	return 0;
}

ccv_nnc_hint_t ccv_nnc_hint_guess(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* inputs, const int input_size, const ccv_nnc_tensor_param_t* outputs, const int output_size)
{
	ccv_nnc_hint_t guess;
	guess.stride.dim[0] = 0;
	guess.border.begin[0] = 0;
	guess.border.end[0] = 0;
	assert(input_size == 1);
	assert(output_size == 1);
	const ccv_nnc_tensor_param_t a = inputs[0];
	const ccv_nnc_tensor_param_t b = outputs[0];
	int i;
	// 0-dim is reserved for channels
	for (i = 1; i < CCV_NNC_MAX_DIM + 1; i++)
	{
		// This is guessed by having a stride that will approximately match the scale.
		int stride = (a.dim[i] + b.dim[i] / 2) / b.dim[i];
		guess.stride.dim[i] = stride;
		int border = (b.dim[i] - 1) * stride - a.dim[i] + cmd.size.dim[i];
		guess.border.begin[i] = border / 2;
		guess.border.end[i] = border - guess.border.begin[i];
	}
	return guess;
}

ccv_nnc_cmd_t ccv_nnc_cmd_autotune(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	// Placeholder yet.
	return cmd;
}

int ccv_nnc_cmd_exec(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size)
{
	assert(cmd.backend < CCV_NNC_BACKEND_COUNT);
	assert(cmd.compute < CCV_NNC_COMPUTE_COUNT);
	// If it is a custom command, just apply it directly.
	if (cmd.compute == CCV_NNC_COMPUTE_CUSTOM)
		return cmd.exec(cmd, hint, flags, inputs, input_size, outputs, output_size);
	ccv_nnc_cmd_api_t api_decl = cmd_api_decls[cmd.backend][cmd.compute];
	int i;
	for (i = 0; i < input_size; i++)
	{
		assert(api_decl.tensor_formats & inputs[i]->info.format);
	}
	for (i = 0; i < output_size; i++)
	{
		assert(api_decl.tensor_formats & outputs[i]->info.format);
	}
	return api_decl.exec(cmd, hint, flags, inputs, input_size, outputs, output_size);
}
