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
		case CCV_NNC_CUSTOM_FORWARD:
			return "CCV_NNC_CUSTOM_FORWARD";
		case CCV_NNC_CUSTOM_BACKWARD:
			return "CCV_NNC_CUSTOM_BACKWARD";
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
	if (backend == CCV_NNC_NO_BACKEND)
		return "CCV_NNC_NO_BACKEND";
	const int idx = _ccv_nnc_cmd_backend_ph(backend);
	assert(idx >= 0);
	assert(idx < CCV_NNC_BACKEND_COUNT);
	return backend_init_map[idx].name;
}

const ccv_nnc_cmd_param_t ccv_nnc_cmd_auto = {};

int ccv_nnc_is_cmd_auto(const ccv_nnc_cmd_param_t params)
{
	return (memcmp(&params, &ccv_nnc_cmd_auto, sizeof(ccv_nnc_cmd_param_t)) == 0);
}

int ccv_nnc_cmd_is_forward(const ccv_nnc_cmd_t cmd)
{
	switch (cmd.cmd)
	{
		case CCV_NNC_NOOP:
			return 0;
		case CCV_NNC_CUSTOM_FORWARD:
		case CCV_NNC_CUSTOM_BACKWARD:
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
		case CCV_NNC_NOOP:
			return 0;
		case CCV_NNC_CUSTOM_FORWARD:
		case CCV_NNC_CUSTOM_BACKWARD:
		case CCV_NNC_GRAPH_FORWARD:
		case CCV_NNC_GRAPH_BACKWARD:
		default:
			return !!(cmd.cmd & 0x1); // If it is odd, it is backward
	}
}

int ccv_nnc_cmd_ok(const uint32_t cmd, const uint32_t backend)
{
	// If it is a custom command, a no op, or a graph op, there is no backend to check.
	if (cmd == CCV_NNC_NOOP ||
		cmd == CCV_NNC_GRAPH_FORWARD || cmd == CCV_NNC_GRAPH_BACKWARD ||
		cmd == CCV_NNC_CUSTOM_FORWARD || cmd == CCV_NNC_CUSTOM_BACKWARD)
		return 1;
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd);
	const int backend_idx = _ccv_nnc_cmd_backend_ph(backend);
	assert(cmd_idx >= 0 && cmd_idx < sizeof(init_map) / sizeof(init_map[0]));
	assert(backend_idx >= 0 && backend_idx < CCV_NNC_BACKEND_COUNT);
	const ccv_nnc_cmd_backend_registry_t api_registry = init_map[cmd_idx].backends[backend_idx];
	// Check if the execution function exists or not.
	return !!api_registry.exec;
}

ccv_nnc_cmd_t ccv_nnc_cmd(const uint32_t _cmd, ccv_nnc_cmd_vtab_t* const isa, const ccv_nnc_cmd_param_t params, const int flags)
{
	ccv_nnc_cmd_t cmd;
	cmd.info = params;
	cmd.backend = CCV_NNC_NO_BACKEND;
	assert((_cmd == CCV_NNC_CUSTOM_FORWARD && isa) || (_cmd != CCV_NNC_CUSTOM_FORWARD && !isa));
	cmd.cmd = _cmd;
	cmd.algorithm = -1; // This is default.
	cmd.isa = isa;
	return cmd;
}

const ccv_nnc_hint_t ccv_nnc_no_hint = {};

int ccv_nnc_is_no_hint(const ccv_nnc_hint_t hint)
{
	return (memcmp(&hint, &ccv_nnc_no_hint, sizeof(ccv_nnc_hint_t)) == 0);
}

int ccv_nnc_hint_verify(const ccv_nnc_hint_t hint, const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t a, const ccv_nnc_tensor_param_t b)
{
	int i;
	assert(a.format == b.format);
	const int nd = ccv_nnc_tensor_nd(a.dim);
	assert(nd == CCV_NNC_MAX_DIM + 1 || nd == CCV_NNC_MAX_DIM + 2);
	int hw;
	if ((a.format == CCV_TENSOR_FORMAT_CHWN) ||
		(a.format == CCV_TENSOR_FORMAT_NHWC && nd == CCV_NNC_MAX_DIM + 1))
		hw = 0;
	else if ((a.format == CCV_TENSOR_FORMAT_NHWC && nd == CCV_NNC_MAX_DIM + 2) ||
			 (a.format == CCV_TENSOR_FORMAT_NCHW && nd == CCV_NNC_MAX_DIM + 1))
		hw = 1;
	else if (a.format == CCV_TENSOR_FORMAT_NCHW && nd == CCV_NNC_MAX_DIM + 2)
		hw = 2;
	else
		assert(0 && "unknown format");
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
	{
		if ((hint.border.begin[i] + hint.border.end[i] + a.dim[i] - cmd.size.dim[i]) % hint.stride.dim[i] != 0)
			return -1;
		int expected = (hint.border.begin[i] + hint.border.end[i] + a.dim[i + hw] - cmd.size.dim[i]) / hint.stride.dim[i] + 1;
		if (expected != b.dim[i + hw])
			return -1;
	}
	return 0;
}

ccv_nnc_hint_t ccv_nnc_hint_auto(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t a, const ccv_nnc_tensor_param_t b)
{
	int i;
	if (a.format != b.format)
		return ccv_nnc_no_hint;
	assert(a.format == b.format);
	const int a_nd = ccv_nnc_tensor_nd(a.dim);
	const int b_nd = ccv_nnc_tensor_nd(b.dim);
	// Is not auto hint deducible dimensions.
	if (a_nd != b_nd || (a_nd != CCV_NNC_MAX_DIM + 1 && a_nd != CCV_NNC_MAX_DIM + 2))
		return ccv_nnc_no_hint;
	int hw;
	if ((a.format == CCV_TENSOR_FORMAT_CHWN) ||
		(a.format == CCV_TENSOR_FORMAT_NHWC && a_nd == CCV_NNC_MAX_DIM + 1))
		hw = 0;
	else if ((a.format == CCV_TENSOR_FORMAT_NHWC && a_nd == CCV_NNC_MAX_DIM + 2) ||
			 (a.format == CCV_TENSOR_FORMAT_NCHW && a_nd == CCV_NNC_MAX_DIM + 1))
		hw = 1;
	else if (a.format == CCV_TENSOR_FORMAT_NCHW && a_nd == CCV_NNC_MAX_DIM + 2)
		hw = 2;
	else
		assert(0 && "unknown format");
	ccv_nnc_hint_t hint_auto = {};
	// 0-dim is reserved for channels
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
	{
		// Cannot have one of the dim is zero, we cannot auto the hint, return no hint.
		assert(a.dim[i + hw] && b.dim[i + hw]);
		// This is guessed by having a stride that will approximately match the scale.
		int stride = (a.dim[i + hw] + b.dim[i + hw] / 2) / b.dim[i + hw];
		hint_auto.stride.dim[i] = stride;
		int border = (b.dim[i + hw] - 1) * stride - a.dim[i + hw] + cmd.size.dim[i];
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
	const ccv_nnc_tensor_param_t z = {};
	int i;
	for (i = 0; i < output_size; i++)
		outputs[i] = z; // Reset the outputs.
	// Cannot handle these situations.
	if (cmd.cmd == CCV_NNC_NOOP || cmd.cmd == CCV_NNC_CUSTOM_FORWARD || cmd.cmd == CCV_NNC_CUSTOM_BACKWARD || cmd.cmd == CCV_NNC_GRAPH_FORWARD || cmd.cmd == CCV_NNC_GRAPH_BACKWARD)
		return;
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	const ccv_nnc_cmd_registry_t registry = init_map[cmd_idx].registry;
	if (registry.tensor_auto)
		registry.tensor_auto(cmd.info, inputs, input_size, hint, outputs, output_size);
	else if (ccv_nnc_cmd_is_forward(cmd)) // For forward, the default auto is forward_from_inputs
		ccv_nnc_hint_tensor_auto_forward_from_inputs(cmd.info, inputs, input_size, hint, outputs, output_size);
	else // For backward, the default auto is backward_from_inputs
		ccv_nnc_hint_tensor_auto_backward_from_inputs(cmd.info, inputs, input_size, hint, outputs, output_size);
}

int ccv_nnc_cmd_allow_inplace(const ccv_nnc_cmd_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	if (cmd.cmd == CCV_NNC_NOOP || cmd.cmd == CCV_NNC_CUSTOM_FORWARD || cmd.cmd == CCV_NNC_CUSTOM_BACKWARD || cmd.cmd == CCV_NNC_GRAPH_FORWARD || cmd.cmd == CCV_NNC_GRAPH_BACKWARD)
		return 0;
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	const ccv_nnc_cmd_registry_t registry = init_map[cmd_idx].registry;
	if (registry.allow_inplace)
		return registry.allow_inplace(input_idx, input_size, output_idx, output_size);
	return 0;
}

int ccv_nnc_cmd_enforce_inplace(const ccv_nnc_cmd_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size)
{
	if (cmd.cmd == CCV_NNC_NOOP || cmd.cmd == CCV_NNC_CUSTOM_FORWARD || cmd.cmd == CCV_NNC_CUSTOM_BACKWARD || cmd.cmd == CCV_NNC_GRAPH_FORWARD || cmd.cmd == CCV_NNC_GRAPH_BACKWARD)
		return 0;
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	const ccv_nnc_cmd_registry_t registry = init_map[cmd_idx].registry;
	if (registry.enforce_inplace)
		return registry.enforce_inplace(input_idx, input_size, output_idx, output_size);
	return 0;
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

uint32_t ccv_nnc_cmd_find_backend(const ccv_nnc_cmd_t cmd, const int tensor_memory, const int tensor_formats, const int tensor_datatypes)
{
	if (cmd.cmd == CCV_NNC_NOOP ||
		cmd.cmd == CCV_NNC_GRAPH_FORWARD || cmd.cmd == CCV_NNC_GRAPH_BACKWARD ||
		cmd.cmd == CCV_NNC_CUSTOM_FORWARD || cmd.cmd == CCV_NNC_CUSTOM_BACKWARD)
		return cmd.backend;
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	assert(cmd_idx >= 0 && cmd_idx < sizeof(init_map) / sizeof(init_map[0]));
	assert(tensor_memory != 0 && tensor_formats != 0 && tensor_datatypes != 0);
	int i;
	for (i = 0; i < CCV_NNC_BACKEND_COUNT; i++)
	{
		const ccv_nnc_cmd_backend_registry_t api_registry = init_map[cmd_idx].backends[i];
		// We have the exec kernel, and support all the tensor memory types.
		if (api_registry.exec &&
			(api_registry.tensor_memory & tensor_memory) == tensor_memory &&
			(api_registry.tensor_formats & tensor_formats) == tensor_formats &&
			(api_registry.tensor_datatypes & tensor_datatypes) == tensor_datatypes)
			return backend_init_map[i].backend;
	}
	return cmd.backend;
}

#define AUTO_TUNE_TRIAL_SIZE (3)

static void _ccv_nnc_cmd_set_device_id(ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
#ifdef HAVE_CUDA
	if (!stream_context)
	{
		int device_id;
		if (ccv_nnc_device_ids_for_io(inputs, input_size, outputs, output_size, &device_id, 1) > 0)
			cudevice(device_id);
	}
#endif
}

ccv_nnc_cmd_t ccv_nnc_cmd_autotune(const ccv_nnc_cmd_t cmd, const size_t max_workspace_size, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// This is a custom cmd kernel, no need to autotune.
	if (cmd.cmd == CCV_NNC_NOOP ||
		cmd.cmd == CCV_NNC_GRAPH_FORWARD || cmd.cmd == CCV_NNC_GRAPH_BACKWARD ||
		cmd.cmd == CCV_NNC_CUSTOM_FORWARD || cmd.cmd == CCV_NNC_CUSTOM_BACKWARD)
		return cmd;
	int i, j, k;
	// Go through all the backends that supports the same type of memory input / output tensors support.
	int tensor_memory = 0, tensor_formats = 0, tensor_datatypes = 0;
	for (i = 0; i < input_size; i++)
		if (inputs[i])
			tensor_memory |= CCV_TENSOR_GET_MEMORY(inputs[i]->info.type), tensor_formats |= inputs[i]->info.format, tensor_datatypes |= inputs[i]->info.datatype;
	for (i = 0; i < output_size; i++)
		if (outputs[i])
			tensor_memory |= CCV_TENSOR_GET_MEMORY(outputs[i]->info.type), tensor_formats |= outputs[i]->info.format, tensor_datatypes |= outputs[i]->info.datatype;
	// In this case, we cannot determine the type of the tensor, skip auto-tune.
	if (!tensor_memory)
		return cmd;
	// Otherwise, we are good to go.
	ccv_nnc_cmd_t tuned_cmd = cmd;
	int64_t best_measured = -1;
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	assert(cmd_idx >= 0 && cmd_idx < sizeof(init_map) / sizeof(init_map[0]));
	int flag = 0;
	for (i = 0; i < CCV_NNC_BACKEND_COUNT; i++)
	{
		const ccv_nnc_cmd_backend_registry_t api_registry = init_map[cmd_idx].backends[i];
		// We have the exec kernel, and support all the tensor memory types.
		if (api_registry.exec &&
			(api_registry.tensor_memory & tensor_memory) == tensor_memory &&
			(api_registry.tensor_formats & tensor_formats) == tensor_formats &&
			(api_registry.tensor_datatypes & tensor_datatypes) == tensor_datatypes)
			if ((++flag) >= 2) // If we have more than 2 suitable backend, we can do this now.
				break;
	}
	if (flag == 0)
		return cmd;
	_ccv_nnc_cmd_set_device_id(inputs, input_size, outputs, output_size, stream_context);
	// Allocate inputs / outputs and fill them in.
	ccv_nnc_tensor_t** const copy_inputs = (ccv_nnc_tensor_t**)malloc(sizeof(ccv_nnc_tensor_t*) * (input_size + output_size * 2));
	ccv_nnc_tensor_t** const copy_outputs = copy_inputs + input_size;
	ccv_nnc_tensor_t** const allocated_outputs = copy_outputs + output_size;
	for (i = 0; i < input_size; i++)
		copy_inputs[i] = (inputs[i]) ? ccv_nnc_tensor_new(0, inputs[i]->info, 0) : 0;
	for (i = 0; i < output_size; i++)
	{
		allocated_outputs[i] = copy_outputs[i] = 0;
		if (outputs[i])
		{
			for (j = 0; j < input_size; j++)
				if (inputs[j])
				{
					if (outputs[i] == inputs[j])
					{
						copy_outputs[i] = copy_inputs[j];
						break;
					} else if (outputs[i]->data.u8 == inputs[j]->data.u8 &&
						ccv_nnc_tensor_count(outputs[i]->info) == ccv_nnc_tensor_count(inputs[j]->info)) {
						allocated_outputs[i] = copy_outputs[i] = ccv_nnc_tensor_new(copy_inputs[j]->data.u8, outputs[i]->info, 0);
						break;
					}
				}
			if (!copy_outputs[i])
				allocated_outputs[i] = copy_outputs[i] = ccv_nnc_tensor_new(0, outputs[i]->info, 0);
		}
	}
	if (flag == 1)
	{
		for (i = 0; i < CCV_NNC_BACKEND_COUNT; i++)
		{
			const ccv_nnc_cmd_backend_registry_t api_registry = init_map[cmd_idx].backends[i];
			// We have the exec kernel, and support all the tensor memory types.
			if (api_registry.exec &&
				(api_registry.tensor_memory & tensor_memory) == tensor_memory &&
				(api_registry.tensor_formats & tensor_formats) == tensor_formats &&
				(api_registry.tensor_datatypes & tensor_datatypes) == tensor_datatypes)
			{
				tuned_cmd.backend = backend_init_map[i].backend;
				// If a given API exist an autotune function, use that to pick the top algorithm.
				if (api_registry.autotune)
				{
					ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, inputs, input_size, copy_inputs, input_size, stream_context);
					_ccv_nnc_cmd_set_device_id(copy_inputs, input_size, copy_outputs, output_size, stream_context);
					tuned_cmd.algorithm = api_registry.autotune(tuned_cmd, max_workspace_size, hint, flags, copy_inputs, input_size, copy_outputs, output_size, stream_context);
					// Drain the context, autotune can use excessive amount of memory. Need to drain it now.
					ccv_nnc_stream_context_drain(stream_context);
				}
				break;
			}
		}
		for (i = 0; i < input_size; i++)
			if (copy_inputs[i])
				ccv_nnc_tensor_free(copy_inputs[i]);
		for (i = 0; i < output_size; i++)
			if (allocated_outputs[i])
				ccv_nnc_tensor_free(allocated_outputs[i]);
		ccfree(copy_inputs);
		return tuned_cmd;
	}
	// We need to have trial loop through all the data.
	for (k = 0; k < AUTO_TUNE_TRIAL_SIZE; k++)
	{
		for (i = 0; i < CCV_NNC_BACKEND_COUNT; i++)
		{
			const ccv_nnc_cmd_backend_registry_t api_registry = init_map[cmd_idx].backends[i];
			// We have the exec kernel, and support all the tensor memory types.
			if (api_registry.exec &&
				(api_registry.tensor_memory & tensor_memory) == tensor_memory &&
				(api_registry.tensor_formats & tensor_formats) == tensor_formats &&
				(api_registry.tensor_datatypes & tensor_datatypes) == tensor_datatypes)
			{
				ccv_nnc_cmd_t candid_cmd = cmd;
				candid_cmd.backend = backend_init_map[i].backend;
				// If a given API exist an autotune function, use that to pick the top algorithm.
				if (api_registry.autotune)
				{
					// Assuming k == 0 is sufficient, and we can skip.
					if (k > 0)
						continue;
					ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, inputs, input_size, copy_inputs, input_size, stream_context);
					_ccv_nnc_cmd_set_device_id(copy_inputs, input_size, copy_outputs, output_size, stream_context);
					candid_cmd.algorithm = api_registry.autotune(candid_cmd, max_workspace_size, hint, flags, copy_inputs, input_size, copy_outputs, output_size, stream_context);
					// Drain the context, autotune can use excessive amount of memory. Need to drain it now.
					ccv_nnc_stream_context_drain(stream_context);
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
						ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, inputs, input_size, copy_inputs, input_size, stream_context);
						_ccv_nnc_cmd_set_device_id(copy_inputs, input_size, copy_outputs, output_size, stream_context);
						uint64_t elapsed = ccv_nnc_cmd_mono_time();
						// Ready to run.
						int status = ccv_nnc_cmd_exec(candid_cmd, hint, flags, copy_inputs, input_size, copy_outputs, output_size, stream_context);
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
	for (i = 0; i < input_size; i++)
		if (copy_inputs[i])
			ccv_nnc_tensor_free(copy_inputs[i]);
	for (i = 0; i < output_size; i++)
		if (allocated_outputs[i])
			ccv_nnc_tensor_free(allocated_outputs[i]);
	ccfree(copy_inputs);
	return tuned_cmd;
}

int ccv_nnc_cmd_bitmask(const ccv_nnc_cmd_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size)
{
	// If it is no-op, return true, it can deal with any number of parameters.
	if (cmd.cmd == CCV_NNC_NOOP)
		return 1;
	// If it is a custom command, I cannot check it at all, return false.
	if (cmd.cmd == CCV_NNC_CUSTOM_FORWARD || cmd.cmd == CCV_NNC_CUSTOM_BACKWARD)
		return 0;
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	const ccv_nnc_cmd_registry_t cmd_registry = init_map[cmd_idx].registry;
	if (cmd_registry.bitmask)
		return cmd_registry.bitmask(input_size, output_size, input_bitmasks, input_bitmask_size, output_bitmasks, output_bitmask_size);
	// If there is not checking, none can pass.
	return 0;
}

int ccv_nnc_device_ids_for_io(ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, int* const device_ids, const int max_device_id_size)
{
	int i, j;
	int device_id_size = 0;
	if (max_device_id_size <= device_id_size)
		return device_id_size;
	// The device id of the exec is determined by its outputs.
	for (i = 0; i < output_size; i++)
		if (outputs[i] &&
			CCV_TENSOR_GET_MEMORY(outputs[i]->info.type) == CCV_TENSOR_GPU_MEMORY &&
			CCV_TENSOR_GET_DEVICE(outputs[i]->info.type) != CCV_COMPUTE_DEVICE_ANY)
		{
			const int device_id = CCV_TENSOR_GET_DEVICE_ID(outputs[i]->info.type);
			int flag = 0;
			for (j = 0; !flag && j < device_id_size; j++)
				flag = (device_ids[j] == device_id);
			if (flag)
				continue;
			device_ids[device_id_size++] = device_id;
			if (device_id_size >= max_device_id_size)
				return device_id_size;
		}
	if (device_id_size == 0)
	{
		int device_id = -1;
		for (i = 0; i < input_size; i++)
			if (inputs[i] &&
				CCV_TENSOR_GET_MEMORY(inputs[i]->info.type) == CCV_TENSOR_GPU_MEMORY &&
				CCV_TENSOR_GET_DEVICE(inputs[i]->info.type) != CCV_COMPUTE_DEVICE_ANY &&
				(device_id < 0 || CCV_TENSOR_GET_DEVICE_ID(inputs[i]->info.type) < device_id))
				device_id = CCV_TENSOR_GET_DEVICE_ID(inputs[i]->info.type);
		if (device_id >= 0)
		{
			device_ids[0] = device_id;
			return 1;
		}
	}
	return device_id_size;
}

int ccv_nnc_cmd_exec(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// If it is no-op, return as if succeed already.
	if (cmd.cmd == CCV_NNC_NOOP)
		return 0;
	_ccv_nnc_cmd_set_device_id(inputs, input_size, outputs, output_size, stream_context);
	// If it is a custom command, just apply it directly.
	if (cmd.cmd == CCV_NNC_CUSTOM_FORWARD || cmd.cmd == CCV_NNC_CUSTOM_BACKWARD)
	{
		int ret = cmd.isa->exec(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
		if (!stream_context)
			ccv_nnc_stream_context_drain(stream_context);
		return ret;
	}
	assert(cmd.cmd != CCV_NNC_GRAPH_FORWARD && cmd.cmd != CCV_NNC_GRAPH_BACKWARD);
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	assert(cmd_idx >= 0 && cmd_idx < sizeof(init_map) / sizeof(init_map[0]));
	int i;
	uint32_t backend = cmd.backend;
	if (backend == CCV_NNC_NO_BACKEND)
	{
		// Find a suitable backend.
		int tensor_memory = 0, tensor_formats = 0, tensor_datatypes = 0;
		for (i = 0; i < input_size; i++)
			if (inputs[i])
				tensor_memory |= CCV_TENSOR_GET_MEMORY(inputs[i]->info.type), tensor_formats |= inputs[i]->info.format, tensor_datatypes |= inputs[i]->info.datatype;
		for (i = 0; i < output_size; i++)
			if (outputs[i])
				tensor_memory |= CCV_TENSOR_GET_MEMORY(outputs[i]->info.type), tensor_formats |= outputs[i]->info.format, tensor_datatypes |= outputs[i]->info.datatype;
		backend = ccv_nnc_cmd_find_backend(cmd, tensor_memory, tensor_formats, tensor_datatypes);
	}
	assert(backend != CCV_NNC_NO_BACKEND);
	const int backend_idx = _ccv_nnc_cmd_backend_ph(backend);
	assert(backend_idx >= 0 && backend_idx < CCV_NNC_BACKEND_COUNT);
	const ccv_nnc_cmd_backend_registry_t api_registry = init_map[cmd_idx].backends[backend_idx];
	if (!api_registry.exec)
		return CCV_NNC_EXEC_NO_KERNEL;
	// Everything is out, call the underlying implementation.
	int ret = api_registry.exec(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
	if (!stream_context)
		ccv_nnc_stream_context_drain(stream_context);
	return ret;
}

int ccv_nnc_cmd_attr(const ccv_nnc_cmd_t cmd, const int flags)
{
	// No additional attr for noop.
	if (cmd.cmd == CCV_NNC_NOOP ||
		// If it is a custom command, just apply it directly.
		cmd.cmd == CCV_NNC_CUSTOM_FORWARD || cmd.cmd == CCV_NNC_CUSTOM_BACKWARD ||
		// If it is sub-graph, there is no additional attr as well.
		cmd.cmd == CCV_NNC_GRAPH_FORWARD || cmd.cmd == CCV_NNC_GRAPH_BACKWARD)
		return 0;
	const int cmd_idx = _ccv_nnc_cmd_ph(cmd.cmd);
	assert(cmd_idx >= 0 && cmd_idx <sizeof(init_map) / sizeof(init_map[0]));
	const ccv_nnc_cmd_registry_t cmd_registry = init_map[cmd_idx].registry;
	return !!(cmd_registry.flags & flags);
}
