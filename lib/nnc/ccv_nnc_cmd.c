#include "ccv_nnc.h"
#include "ccv_nnc_internal.h"
#include "ccv_nnc_easy.h"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#endif
#include <time.h>
#include <sys/time.h>

#ifdef __MACH__
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif

typedef struct {
	const uint32_t cmd;
	const char* name;
	ccv_nnc_cmd_registry_t registry;
	ccv_nnc_cmd_backend_registry_t backends[CCV_NNC_BACKEND_COUNT];
} ccv_nnc_cmd_init_t;

typedef struct {
	const uint32_t backend;
	const char* name;
} ccv_nnc_cmd_backend_init_t;

// The generated code configures command and its mapping.
#include "cmd/ccv_nnc_cmd.inc"

void ccv_nnc_init(void)
{
	_ccv_nnc_cmd_init();
}

const char* ccv_nnc_cmd_name(const uint32_t cmd)
{
	switch (cmd)
	{
		case CCV_NNC_NOOP:
			return "CCV_NNC_NOOP";
		case CCV_NNC_CUSTOM:
			return "CCV_NNC_CUSTOM";
		case CCV_NNC_GRAPH_FORWARD:
			return "CCV_NNC_GRAPH_FORWARD";
		case CCV_NNC_GRAPH_BACKWARD:
			return "CCV_NNC_GRAPH_BACKWARD";
	}
	const int idx = _ccv_nnc_cmd_ph(cmd);
	assert(idx >= 0);
	assert(idx < sizeof(init_map) / sizeof(init_map[0]));
	return init_map[idx].name;
}

const char* ccv_nnc_cmd_backend_name(const uint32_t backend)
{
	const int idx = _ccv_nnc_cmd_backend_ph(backend);
	assert(idx >= 0);
	assert(idx < CCV_NNC_BACKEND_COUNT);
	return backend_init_map[idx].name;
}

const ccv_nnc_cmd_param_t ccv_nnc_cmd_auto = {{{0}}};

int ccv_nnc_is_cmd_auto(const ccv_nnc_cmd_param_t params)
{
	return (memcmp(&params, &ccv_nnc_cmd_auto, sizeof(ccv_nnc_cmd_param_t)) == 0);
}

int ccv_nnc_cmd_is_forward(const ccv_nnc_cmd_t cmd)
{
	switch (cmd.cmd)
	{
		case CCV_NNC_CUSTOM:
		case CCV_NNC_NOOP:
			return 0;
		case CCV_NNC_GRAPH_FORWARD:
		case CCV_NNC_GRAPH_BACKWARD:
		default:
			return !(cmd.cmd & 0x1); // If it is even, it is forward
	}
}

int ccv_nnc_cmd_is_backward(const ccv_nnc_cmd_t cmd)
{
	switch (cmd.cmd)
	{
		case CCV_NNC_CUSTOM:
		case CCV_NNC_NOOP:
			return 0;
		case CCV_NNC_GRAPH_FORWARD:
		case CCV_NNC_GRAPH_BACKWARD:
		default:
			return !!(cmd.cmd & 0x1); // If it is odd, it is backward
	}
}

ccv_nnc_cmd_t ccv_nnc_cmd(const uint32_t _cmd, ccv_nnc_cmd_exec_f exec, const ccv_nnc_cmd_param_t params, const int flags)
{
	ccv_nnc_cmd_t cmd;
	cmd.info = params;
	// Default to CPU ref implementation if the type is CPU memory, otherwise use GPU ref.
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert((_cmd == CCV_NNC_CUSTOM && exec) || (_cmd != CCV_NNC_CUSTOM && !exec));
	cmd.cmd = _cmd;
	cmd.algorithm = -1; // This is default.
	cmd.exec = exec;
	return cmd;
}

const ccv_nnc_hint_t ccv_nnc_no_hint = {{{0}}};

int ccv_nnc_is_no_hint(const ccv_nnc_hint_t hint)
{
	return (memcmp(&hint, &ccv_nnc_no_hint, sizeof(ccv_nnc_hint_t)) == 0);
}

int ccv_nnc_hint_verify(const ccv_nnc_hint_t hint, const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t a, const ccv_nnc_tensor_param_t b)
{
	int i;
	assert(CCV_TENSOR_GET_FORMAT(a.format) == CCV_TENSOR_GET_FORMAT(b.format));
	const int hw = (CCV_TENSOR_GET_FORMAT(a.format) == CCV_TENSOR_FORMAT_CHWN || CCV_TENSOR_GET_FORMAT(a.format) == CCV_TENSOR_FORMAT_NHWC) ? 1 : 0;
	for (i = hw; i < CCV_NNC_MAX_DIM + hw; i++)
	{
		if ((hint.border.begin[i] + hint.border.end[i] + a.dim[i] - cmd.size.dim[i]) % hint.stride.dim[i] != 0)
			return -1;
		int expected = (hint.border.begin[i] + hint.border.end[i] + a.dim[i] - cmd.size.dim[i]) / hint.stride.dim[i] + 1;
		if (expected != b.dim[i])
			return -1;
	}
	return 0;
}

ccv_nnc_hint_t ccv_nnc_hint_auto(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t a, const ccv_nnc_tensor_param_t b)
{
	int i;
	assert(CCV_TENSOR_GET_FORMAT(a.format) == CCV_TENSOR_GET_FORMAT(b.format));
	const int hw = (CCV_TENSOR_GET_FORMAT(a.format) == CCV_TENSOR_FORMAT_CHWN || CCV_TENSOR_GET_FORMAT(a.format) == CCV_TENSOR_FORMAT_NHWC) ? 1 : 0;
	for (i = hw; i < CCV_NNC_MAX_DIM + hw; i++)
		if (!a.dim[i] || !b.dim[i]) // If one of the dim is zero, we cannot auto the hint, return no hint.
			return ccv_nnc_no_hint;
	ccv_nnc_hint_t hint_auto = {
		.stride = {
			.dim = {0}
		},
		.border = {
			.begin = {0},
			.end = {0}
		}
	};
	// 0-dim is reserved for channels
	for (i = hw; i < CCV_NNC_MAX_DIM + hw; i++)
	{
		// This is guessed by having a stride that will approximately match the scale.
		int stride = (a.dim[i] + b.dim[i] / 2) / b.dim[i];
		hint_auto.stride.dim[i] = stride;
		int border = (b.dim[i] - 1) * stride - a.dim[i] + cmd.size.dim[i];
		hint_auto.border.begin[i] = (border + 1) / 2; // Always prefer to have more padding in the beginning, this matches CUDNN behavior.
		hint_auto.border.end[i] = border - hint_auto.border.begin[i];
	}
	return hint_auto;
}

void ccv_nnc_hint_tensor_auto_forward_from_inputs(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	int i;
	assert(output_size <= input_size);
	for (i = 0; i < output_size; i++)
		outputs[i] = inputs[i];
}

void ccv_nnc_hint_tensor_auto_backward_from_gradient(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	int i;
	for (i = 0; i < output_size; i++)
		outputs[i] = inputs[0];
}

void ccv_nnc_hint_tensor_auto_backward_from_inputs(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	int i;
	assert(output_size < input_size);
	for (i = 0; i < output_size; i++)
		outputs[i] = inputs[i + 1];
}

void ccv_nnc_hint_tensor_auto(const ccv_nnc_cmd_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size)
{
	// zero out the parameters
	const ccv_nnc_tensor_param_t z = {
		.type = 0,
		.format = 0,
		.dim = {0}
	};
	int i;
	for (i = 0; i < output_size; i++)
		outputs[i] = z; // Reset the outputs.
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	const ccv_nnc_cmd_registry_t registry = init_map[cmd_idx].registry;
	if (registry.tensor_auto)
		registry.tensor_auto(cmd.info, inputs, input_size, hint, outputs, output_size);
	else if (ccv_nnc_cmd_is_forward(cmd)) // For forward, the default auto is forward_from_inputs
		ccv_nnc_hint_tensor_auto_forward_from_inputs(cmd.info, inputs, input_size, hint, outputs, output_size);
	else // For backward, the default auto is backward_from_inputs
		ccv_nnc_hint_tensor_auto_backward_from_inputs(cmd.info, inputs, input_size, hint, outputs, output_size);
}

// This returns absolute time.
uint64_t ccv_nnc_cmd_mono_time(void)
{
#ifdef __MACH__
	return mach_absolute_time();
#else
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
#endif
}

#define AUTO_TUNE_TRIAL_SIZE (3)

ccv_nnc_cmd_t ccv_nnc_cmd_autotune(const ccv_nnc_cmd_t cmd, const size_t max_workspace_size, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	// This is a custom cmd kernel, no need to autotune.
	if (cmd.cmd == CCV_NNC_CUSTOM)
		return cmd;
	int i, j, k;
	// Go through all the backends that supports the same type of memory input / output tensors support.
	int tensor_memory = 0, tensor_formats = 0;
	for (i = 0; i < input_size; i++)
		if (inputs[i])
			tensor_memory |= inputs[i]->info.type, tensor_formats |= inputs[i]->info.format;
	for (i = 0; i < output_size; i++)
		if (outputs[i])
			tensor_memory |= outputs[i]->info.type, tensor_formats |= inputs[i]->info.format;
	// In this case, we cannot determine the type of the tensor, skip auto-tune.
	if (!tensor_memory)
		return cmd;
	// Otherwise, we are good to go.
	ccv_nnc_cmd_t tuned_cmd = cmd;
	int64_t best_measured = -1;
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	// We need to have trial loop through all the data.
	for (k = 0; k < AUTO_TUNE_TRIAL_SIZE; k++)
	{
		for (i = 0; i < CCV_NNC_BACKEND_COUNT; i++)
		{
			const ccv_nnc_cmd_backend_registry_t api_registry = init_map[cmd_idx].backends[i];
			// We have the exec kernel, and support all the tensor memory types.
			if (api_registry.exec &&
				(api_registry.tensor_memory & tensor_memory) == tensor_memory &&
				(api_registry.tensor_formats & tensor_formats) == tensor_formats)
			{
				ccv_nnc_cmd_t candid_cmd = cmd;
				candid_cmd.backend = backend_init_map[i].backend;
				// If a given API exist an autotune function, use that to pick the top algorithm.
				if (api_registry.autotune)
				{
					// Assuming k == 0 is sufficient, and we can skip.
					if (k > 0)
						continue;
					candid_cmd.algorithm = api_registry.autotune(candid_cmd, max_workspace_size, hint, flags, inputs, input_size, outputs, output_size, stream_context);
					uint64_t elapsed = ccv_nnc_cmd_mono_time();
					// Ready to run.
					int status = ccv_nnc_cmd_exec(candid_cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
					ccv_nnc_stream_context_wait(stream_context);
					elapsed = ccv_nnc_cmd_mono_time() - elapsed;
					if (status == CCV_NNC_EXEC_SUCCESS &&
						(best_measured == -1 || elapsed < best_measured))
					{
						best_measured = elapsed;
						tuned_cmd = candid_cmd;
					}
				} else {
					// Otherwise loop over the existing algorithms and pick the top one.
					for (j = 0; j < api_registry.algorithms; j++)
					{
						candid_cmd.algorithm = j;
						uint64_t elapsed = ccv_nnc_cmd_mono_time();
						// Ready to run.
						int status = ccv_nnc_cmd_exec(candid_cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
						elapsed = ccv_nnc_cmd_mono_time() - elapsed;
						if (status == CCV_NNC_EXEC_SUCCESS &&
							(best_measured == -1 || elapsed < best_measured))
						{
							best_measured = elapsed;
							tuned_cmd = candid_cmd;
						}
					}
				}
			}
		}
	}
	return tuned_cmd;
}

int ccv_nnc_cmd_bitmask(const ccv_nnc_cmd_t cmd, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// If it is no-op, return true, it can deal with any number of parameters.
	if (cmd.cmd == CCV_NNC_NOOP)
		return 1;
	// If it is a custom command, I cannot check it at all, return true.
	if (cmd.cmd == CCV_NNC_CUSTOM)
		return 1;
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	const ccv_nnc_cmd_registry_t cmd_registry = init_map[cmd_idx].registry;
	if (cmd_registry.bitmask)
		return cmd_registry.bitmask(input_bitmasks, input_bitmask_size, output_bitmasks, output_bitmask_size);
	// If there is not checking, none can pass.
	return 0;
}

int ccv_nnc_cmd_exec(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	// If it is no-op, return as if succeed already.
	if (cmd.cmd == CCV_NNC_NOOP)
		return 0;
	// If it is a custom command, just apply it directly.
	if (cmd.cmd == CCV_NNC_CUSTOM)
		return cmd.exec(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
	assert(cmd.cmd != CCV_NNC_GRAPH_FORWARD && cmd.cmd != CCV_NNC_GRAPH_BACKWARD);
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	const int backend_idx = _ccv_nnc_cmd_backend_ph(cmd.backend);
	assert(cmd_idx >= 0 && cmd_idx < sizeof(init_map) / sizeof(init_map[0]));
	assert(backend_idx >= 0 && backend_idx < CCV_NNC_BACKEND_COUNT);
	const ccv_nnc_cmd_registry_t cmd_registry = init_map[cmd_idx].registry;
	const ccv_nnc_cmd_backend_registry_t api_registry = init_map[cmd_idx].backends[backend_idx];
	if (!api_registry.exec)
		return CCV_NNC_EXEC_NO_KERNEL;
	int i;
	uint64_t stack_input_bitmasks[CCV_NNC_STACK_BITMASK_ALLOC] = {0};
	uint64_t stack_output_bitmasks[CCV_NNC_STACK_BITMASK_ALLOC] = {0};
	assert(CCV_NNC_STACK_BITMASK_ALLOC > 0);
	uint64_t* input_bitmasks = (input_size > 64 * CCV_NNC_STACK_BITMASK_ALLOC) ? (uint64_t*)cccalloc((input_size + 63) / 64, sizeof(uint64_t)) : stack_input_bitmasks;
	uint64_t* output_bitmasks = (output_size > 64 * CCV_NNC_STACK_BITMASK_ALLOC) ? (uint64_t*)cccalloc((input_size + 63) / 64, sizeof(uint64_t)) : stack_output_bitmasks;
	for (i = 0; i < input_size; i++)
		if (inputs[i])
		{
			assert(CCV_GET_DATA_TYPE(api_registry.tensor_formats) & CCV_GET_DATA_TYPE(inputs[i]->info.format));
			assert(CCV_TENSOR_GET_FORMAT(api_registry.tensor_formats) & CCV_TENSOR_GET_FORMAT(inputs[i]->info.format));
			input_bitmasks[i / 64] |= (uint64_t)1 << i;
		}
	for (i = 0; i < output_size; i++)
		if (outputs[i])
		{
			assert(CCV_GET_DATA_TYPE(api_registry.tensor_formats) & CCV_GET_DATA_TYPE(outputs[i]->info.format));
			assert(CCV_TENSOR_GET_FORMAT(api_registry.tensor_formats) & CCV_TENSOR_GET_FORMAT(outputs[i]->info.format));
			output_bitmasks[i / 64] |= (uint64_t)1 << i;
		}
	if (cmd_registry.bitmask)
		// If cannot pass the bitmask check.
		if (!cmd_registry.bitmask(input_bitmasks, (input_size + 63) / 64, output_bitmasks, (output_size + 63) / 64))
		{
			if (input_size > 64 * CCV_NNC_STACK_BITMASK_ALLOC)
				ccfree(input_bitmasks);
			if (output_size > 64 * CCV_NNC_STACK_BITMASK_ALLOC)
				ccfree(output_bitmasks);
			return CCV_NNC_EXEC_INVALID; // Return invalid input.
		}
	// TODO: Print out warning message.
	if (input_size > 64 * CCV_NNC_STACK_BITMASK_ALLOC)
		ccfree(input_bitmasks);
	if (output_size > 64 * CCV_NNC_STACK_BITMASK_ALLOC)
		ccfree(output_bitmasks);
	// Everything is out, call the underlying implementation.
	return api_registry.exec(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
}

int ccv_nnc_cmd_attr(const ccv_nnc_cmd_t cmd, const int flags)
{
	// If it is a custom command, just apply it directly.
	assert(cmd.cmd != CCV_NNC_CUSTOM);
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	const int backend_idx = _ccv_nnc_cmd_backend_ph(cmd.backend);
	assert(cmd_idx >= 0 && cmd_idx <sizeof(init_map) / sizeof(init_map[0]));
	assert(backend_idx >= 0 && backend_idx < CCV_NNC_BACKEND_COUNT);
	const ccv_nnc_cmd_registry_t cmd_registry = init_map[cmd_idx].registry;
	return !!(cmd_registry.flags & flags);
}

struct ccv_nnc_stream_context_s {
	int type;
	// Left for implementation yet, the CPU support for stream context.
};

ccv_nnc_stream_context_t* ccv_nnc_stream_context_new(const int type)
{
	ccv_nnc_stream_context_t* stream_context = (ccv_nnc_stream_context_t*)ccmalloc(sizeof(ccv_nnc_stream_context_t));
	stream_context->type = type;
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(type) == CCV_STREAM_CONTEXT_GPU)
		stream_context = ccv_nnc_init_stream_context(stream_context);
#endif
	return stream_context;
}

void ccv_nnc_stream_context_wait(const ccv_nnc_stream_context_t* const stream_context)
{
	if (!stream_context)
		return;
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(stream_context->type) == CCV_STREAM_CONTEXT_GPU)
		ccv_nnc_synchronize_stream_context(stream_context);
#endif
}

void ccv_nnc_stream_context_free(ccv_nnc_stream_context_t* const stream_context)
{
#ifdef HAVE_CUDA
	if (CCV_STREAM_GET_CONTEXT(stream_context->type) == CCV_STREAM_CONTEXT_GPU)
		ccv_nnc_deinit_stream_context(stream_context);
#endif
	ccfree(stream_context);
}
